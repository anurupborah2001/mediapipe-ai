
## Import the images
import cv2
import numpy as np

from module.hand_detector import HandDetector
from openvisionkit.capture.video_template import KeyEventManager, video_capture_template
from util import utility


path, _= utility.get_calling_folder()
background_img =  utility.load_image(f"{path}/images/background.png", False)
gameover_img = utility.load_image(f"{path}/images/game_over.png")
ping_pong_ball = utility.load_image(f"{path}/images/ball.png")
ping_pong_paddle_left = utility.load_image(f"{path}/images/paddle1.png")
ping_pong_paddle_right = utility.load_image(f"{path}/images/paddle2.png")

hand_detector = HandDetector(detection_confidence=0.8)
game_state = {
    "ball_pos": [100, 100], 
    "speed_x": 10,
    "speed_y": 10,
    "game_over": False,
    "score": [0, 0]
}

key_manager = KeyEventManager()

def reset_game(frame, state):
   global game_state, gameover_img
   game_state["ball_pos"] = [100, 100]
   game_state["speed_x"] = 15
   game_state["speed_y"] = 15
   game_state["game_over"] = False
   game_state["score"] = [0, 0]

   gameover_img = cv2.imread(
       f"{path}/images/game_over.png"
   )
   print("Game Reset")
   
key_manager.register(ord("r"), reset_game)

def ping_pong_game_logic(frame):
  global game_state, gameover_img
  # Placeholder for ping pong game logic
  frame = cv2.flip(frame, 1) ## Horizontal flip for mirror effect
  annotated_image, landmarks = hand_detector.draw_landmarks(frame, flip_hands=True) ## Flip hand labels for mirror effect
  # landmarks_list, hand_bounding_box, landmark_params = landmarks
  annotated_image = cv2.addWeighted(annotated_image, 0.2, background_img, 0.8, 0)
  if len(landmarks) != 0:
    for hand_landmark in landmarks:
      landmarks_list, bounding_box, landmark_params, hand_label  = hand_landmark
      ##Bounding box coordinates
      x, y, w, h = bounding_box
      height_paddle, width_paddle, _ = ping_pong_paddle_left.shape
      y1 = y - height_paddle // 2
      y1 = np.clip(y1, 20, 555)
      if hand_label == "Left":
        annotated_image = utility.overlay_transparent(annotated_image, ping_pong_paddle_left, (60, y1))
        if 60 < game_state["ball_pos"][0] < 60 + width_paddle and y1 < game_state["ball_pos"][1] < y1 + height_paddle:
          game_state["speed_x"] = -game_state["speed_x"]
          game_state["score"][0] += 1
          print("Left player scored! Score:", game_state["score"])
      if hand_label == "Right":
        annotated_image = utility.overlay_transparent(annotated_image, ping_pong_paddle_right, (1195, y1))
        if 1195 < game_state["ball_pos"][0] < 1195 + width_paddle and y1 < game_state["ball_pos"][1] < y1 + height_paddle:
          game_state["speed_x"] = -game_state["speed_x"]
          game_state["score"][1] += 1
          print("Right player scored! Score:", game_state["score"])
          
  if game_state["ball_pos"][0] <= 20:
    game_state["game_over"] = True
    print("Right player wins!")
  elif game_state["ball_pos"][0] >= 1200:
    game_state["game_over"] = True
    print("Left player wins!")
        
  if game_state["game_over"]:
    annotated_image = gameover_img
    cv2.putText(annotated_image, str(game_state["score"][1] + game_state["score"][0]).zfill(2), (580, 340), cv2.FONT_HERSHEY_COMPLEX,
                    2.5, (200, 0, 200), 5)
  else:
    # Move the ball
    if game_state["ball_pos"][1] >= 560 or game_state["ball_pos"][1] <= 20:
        game_state["speed_y"] = -game_state["speed_y"]

    game_state["ball_pos"][0] += game_state["speed_x"]
    game_state["ball_pos"][1] += game_state["speed_y"]    
    
    annotated_image = utility.overlay_transparent(annotated_image, ping_pong_ball, (game_state["ball_pos"][0], game_state["ball_pos"][1]))
    ##Score for the 1st player is displayed on the left side and score for the 2nd player is displayed on the right side of the screen. The scores are updated in real-time as the game progresses, allowing players to keep track of their performance during the ping pong game.
    cv2.putText(annotated_image, str(game_state["score"][0]), (300, 650), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 165, 185), 5)
    cv2.putText(annotated_image, str(game_state["score"][1]), (900, 650), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 165, 185), 5)
        
    key = cv2.waitKey(1)
    if key == ord('r'):
      game_state["ball_pos"] = [100, 100]
      game_state["speed_x"] = 15
      game_state["speed_y"] = 15
      game_state["game_over"] = False
      game_state["score"]  = [0, 0]
      gameover_img = utility.load_image(f"{path}/images/game_over.png")
    # Get the coordinates of the hand landmarks
    # hand_landmark = landmarks_list[0]
    # hand_center_x, hand_center_y = hand_landmark[9][1], hand_landmark[9][2]  # Using landmark 9 (middle finger MCP) as reference point
    
    # if 
    # # Draw the ping pong paddle at the hand position
    # paddle_width, paddle_height, _ = ping_pong_paddle1.shape
    # paddle_x = hand_center_x - paddle_width // 2
    # paddle_y = hand_center_y - paddle_height // 2
    # annotated_image[paddle_y:paddle_y + paddle_height, paddle_x:paddle_x + paddle_width] = ping_pong_paddle1
   
  return annotated_image

if __name__ == "__main__":

  video_capture_template(
    custom_logic=ping_pong_game_logic,
    window_name="Kids Game - Ping Pong",
    state=game_state,
    key_manager=key_manager,
    draw_fps=False,
    enable_screenshot=True
  )
