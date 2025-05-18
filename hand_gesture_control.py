import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Get screen size
screen_width, screen_height = pyautogui.size()

# Initialize variables
prev_x, prev_y = 0, 0
curr_x, curr_y = 0, 0
smooth_factor = 0.5  # Adjust this value to change mouse movement smoothness
click_radius = 0  # For click animation
click_color = (0, 255, 0)  # Green color for click indicator
last_click_time = 0
click_cooldown = 0.5  # Cooldown time between clicks in seconds

def calculate_distance(p1, p2):
    """Calculate distance between two points"""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def map_coordinates(x, y, frame_width, frame_height):
    """Map hand coordinates to screen coordinates"""
    # Map x from [0, frame_width] to [0, screen_width]
    screen_x = int(np.interp(x, [0, frame_width], [0, screen_width]))
    # Map y from [0, frame_height] to [0, screen_height]
    screen_y = int(np.interp(y, [0, frame_height], [0, screen_height]))
    return screen_x, screen_y

def draw_click_animation(frame, x, y, radius):
    """Draw click animation circle"""
    cv2.circle(frame, (x, y), radius, click_color, 2)
    return radius + 2 if radius < 30 else 0

def count_fingers(hand_landmarks):
    """Count the number of fingers that are up"""
    finger_tips = [
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP
    ]
    
    finger_mcps = [
        mp_hands.HandLandmark.INDEX_FINGER_MCP,
        mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
        mp_hands.HandLandmark.RING_FINGER_MCP,
        mp_hands.HandLandmark.PINKY_MCP
    ]
    
    fingers_up = 0
    for tip, mcp in zip(finger_tips, finger_mcps):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[mcp].y:
            fingers_up += 1
            
    return fingers_up

def main():
    global prev_x, prev_y, curr_x, curr_y, last_click_time, click_radius
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            break
            
        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)
        
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and detect hands
        results = hands.process(rgb_frame)
        
        # Get frame dimensions
        frame_height, frame_width, _ = frame.shape
        
        # Draw click animation
        if click_radius > 0:
            click_radius = draw_click_animation(frame, curr_x, curr_y, click_radius)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_draw.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS
                )
                
                # Get index finger tip coordinates for mouse movement
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                
                # Convert normalized coordinates to pixel values
                index_x = int(index_tip.x * frame_width)
                index_y = int(index_tip.y * frame_height)
                
                # Map coordinates to screen
                screen_x, screen_y = map_coordinates(index_x, index_y, frame_width, frame_height)
                
                # Smooth mouse movement
                curr_x = int(prev_x + (screen_x - prev_x) * smooth_factor)
                curr_y = int(prev_y + (screen_y - prev_y) * smooth_factor)
                
                # Move mouse
                pyautogui.moveTo(curr_x, curr_y)
                
                # Update previous coordinates
                prev_x, prev_y = curr_x, curr_y
                
                # Count fingers
                fingers_up = count_fingers(hand_landmarks)
                current_time = time.time()
                
                # Perform actions based on number of fingers
                if fingers_up == 1 and (current_time - last_click_time) > click_cooldown:
                    # Hover mode - no action needed
                    cv2.putText(frame, "Hover Mode", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                elif fingers_up == 2 and (current_time - last_click_time) > click_cooldown:
                    # Left click
                    pyautogui.click()
                    last_click_time = current_time
                    click_radius = 5  # Start click animation
                    cv2.putText(frame, "Left Click!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                elif fingers_up == 3 and (current_time - last_click_time) > click_cooldown:
                    # Right click
                    pyautogui.rightClick()
                    last_click_time = current_time
                    click_radius = 5  # Start click animation
                    cv2.putText(frame, "Right Click!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Add instructions to the frame
        cv2.putText(frame, "1 Finger: Hover", (10, frame_height - 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "2 Fingers: Left Click", (10, frame_height - 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "3 Fingers: Right Click", (10, frame_height - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display the frame
        cv2.imshow('Hand Gesture Control', frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 