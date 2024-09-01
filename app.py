from flask import Flask, request, jsonify, render_template, Response, session
from flask_wtf import CSRFProtect
import openai
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time
import os

app = Flask(__name__)
csrf = CSRFProtect(app)

app.secret_key = 'default_secret_key'


# 모델 로드
right_model = tf.keras.models.load_model('models/best_right_model.keras')
left_model = tf.keras.models.load_model('models/best_left_model.keras')

# 동작 정의
actions_right = ['meet', 'nice', 'hello', 'you', 'name', 'what', 'have', 'do not have', 'me']
actions_left = ['meet', 'nice', 'hello', 'you', 'name', 'what', 'have', 'do not have', 'me']
actions_both = ['meet', 'nice', 'hello', 'you', 'name', 'what', 'have', 'do not have', 'me']

# 시퀀스 길이
seq_length = 30

# MediaPipe Hands 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def get_file_path(base_folder, user_id, filename):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    user_dir = os.path.join(base_dir, base_folder, user_id)
    os.makedirs(user_dir, exist_ok=True)
    file_path = os.path.join(user_dir, filename)
    return file_path

def get_user_file_path(user_id):
    return get_file_path('data', user_id, 'example.txt')

def get_result_file_path(user_id):
    return get_file_path('result', user_id, 'example.txt')

def append_to_file(file_path, text):
    try:
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(f"{text}\n")
    except IOError as e:
        print(f"Error appending to file: {e}")

def read_text_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except IOError as e:
        print(f"Error reading file: {e}")
        return None

def get_completion(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.5,
        )
        message = response.choices[0].message['content']
        return message
    except openai.error.OpenAIError as e:
        if e.code == 'quota_exceeded':
            return "You have exceeded your quota. Please check your OpenAI plan and billing details."
        else:
            print(f"Error occurred: {e}")
            return "There was an error processing your request."

@app.route('/translate')
def translate_view():
    return render_template('translate.html')

@app.route('/video_feed')
def video_feed():
    def gen_frames(user_id):
        sequence = {'left': [], 'right': []}
        action_seq = []
        last_action_time = 0
        this_action = ''
        last_saved_action = ''

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap = cv2.VideoCapture(1)

        while cap.isOpened():
            ret, img = cap.read()
            if not ret:
                break

            img = cv2.flip(img, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(img_rgb)
            current_time = time.time()

            if result.multi_hand_landmarks:
                for hand_landmarks, hand_info in zip(result.multi_hand_landmarks, result.multi_handedness):
                    joint = np.zeros((21, 4))
                    for j, lm in enumerate(hand_landmarks.landmark):
                        joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                    v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :3]
                    v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :3]
                    v = v2 - v1
                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                    dot_product = np.einsum('nt,nt->n', v, v)
                    dot_product = np.clip(dot_product, -1.0, 1.0)

                    angle = np.arccos(dot_product)
                    angle = np.degrees(angle)

                    d = np.concatenate([joint.flatten(), angle])

                    hand_label = hand_info.classification[0].label
                    if hand_label == 'Right':
                        sequence['right'].append(d)
                        if len(sequence['right']) > seq_length:
                            sequence['right'].pop(0)
                    else:
                        sequence['left'].append(d)
                        if len(sequence['left']) > seq_length:
                            sequence['left'].pop(0)

                if len(sequence['right']) == seq_length and len(sequence['left']) == seq_length:
                    input_data_right = np.expand_dims(np.array(sequence['right']), axis=0)
                    input_data_left = np.expand_dims(np.array(sequence['left']), axis=0)

                    y_pred_right = right_model.predict(input_data_right).squeeze()
                    y_pred_left = left_model.predict(input_data_left).squeeze()

                    i_pred_right = int(np.argmax(y_pred_right))
                    i_pred_left = int(np.argmax(y_pred_left))

                    conf_right = y_pred_right[i_pred_right]
                    conf_left = y_pred_left[i_pred_left]
                    
                    if i_pred_right < len(actions_right):
                        print(f"Right hand prediction: {actions_right[i_pred_right]} ({conf_right:.2f})")
                    else:
                        print(f"Right hand prediction index {i_pred_right} is out of range for actions_right")

                    if i_pred_left < len(actions_left):
                        print(f"Left hand prediction: {actions_left[i_pred_left]} ({conf_left:.2f})")
                    else:
                        print(f"Left hand prediction index {i_pred_left} is out of range for actions_left")

                    if conf_right > 0.5 and conf_left > 0.5:
                        if i_pred_right < len(actions_both) and i_pred_left < len(actions_both):
                            action = actions_both[i_pred_right]
                            action_seq.append(action)

                            if len(action_seq) > 3:
                                action_seq = action_seq[-3:]

                            if action_seq.count(action) > 1:
                                this_action = action
                            else:
                                this_action = ' '

                            last_action_time = current_time
                            sequence = {'left': [], 'right': []}
                
                elif len(sequence['right']) == seq_length:
                    input_data = np.expand_dims(np.array(sequence['right']), axis=0)
                    y_pred = right_model.predict(input_data).squeeze()
                    i_pred = int(np.argmax(y_pred))
                    conf = y_pred[i_pred]
                    
                    if i_pred < len(actions_right):
                        print(f"Right hand prediction: {actions_right[i_pred]} ({conf:.2f})")

                    if conf > 0.5:
                        action = actions_right[i_pred]
                        action_seq.append(action)

                        if len(action_seq) > 3:
                            action_seq = action_seq[-3:]

                        if action_seq.count(action) > 1:
                            this_action = action
                        else:
                            this_action = ' '

                        last_action_time = current_time
                        sequence = {'left': [], 'right': []}
                        
                    else:
                        print(f"Right hand prediction index {i_pred} is out of range for actions_right")

                elif len(sequence['left']) == seq_length:
                    input_data = np.expand_dims(np.array(sequence['left']), axis=0)
                    y_pred = left_model.predict(input_data).squeeze()
                    i_pred = int(np.argmax(y_pred))
                    conf = y_pred[i_pred]
                    
                    if i_pred < len(actions_left):
                        print(f"Left hand prediction: {actions_left[i_pred]} ({conf:.2f})")

                    if conf > 0.5:
                        action = actions_left[i_pred]
                        action_seq.append(action)

                        if len(action_seq) > 3:
                            action_seq = action_seq[-3:]

                        if action_seq.count(action) > 1:
                            this_action = action
                        else:
                            this_action = ' '

                        last_action_time = current_time
                        sequence = {'left': [], 'right': []}
                        
                    else:
                        print(f"Left hand prediction index {i_pred} is out of range for actions_left")

                for hand_landmarks in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if current_time - last_action_time < 1:
                cv2.putText(img, this_action, org=(0, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

                if this_action != last_saved_action:
                    file_path = get_user_file_path(user_id)
                    append_to_file(file_path, this_action)
                    last_saved_action = this_action

            ret, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        cap.release()

    user_id = session.get('user_id')
    if not user_id:
        session['user_id'] = user_id = str(time.time())

    return Response(gen_frames(user_id), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/', methods=['GET', 'POST'])
def query_view():
    if request.method == 'POST':
        prompt = request.form.get('prompt')
        user_id = session.get('user_id')
        if not user_id:
            user_id = str(time.time())
            session['user_id'] = user_id

        file_path = get_user_file_path(user_id)
        result_file_path = get_result_file_path(user_id)
        print(f"파일 경로: {file_path}")

        if not os.path.exists(file_path):
            print("파일이 존재하지 않습니다.")
            return jsonify({'response': "파일을 찾을 수 없습니다."})

        content = read_text_file(file_path)
        if content is None:
            return jsonify({'response': "파일을 읽을 수 없습니다."})

        content += "\n지금까지 나온 단어들을 중복된 단어가 있다면 한 단어만 사용해서 한글로 번역 후 자연스러운 문장으로 만들어줘"
        response = get_completion(content)
        append_to_file(result_file_path, response)

        return jsonify({'response': response})
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
