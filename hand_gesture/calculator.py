import cv2
import time
import ast
import operator as op

from module.hand_detector import HandDetector
from openvisionkit.capture.video_template import video_capture_template
from util import utility


hand_detector = HandDetector(max_hands=1, detection_confidence=0.7)

mouse_x, mouse_y = 0, 0
mouse_clicked = False

calculator_text = ""
last_click_time = 0
CLICK_DELAY = 0.45


def mouse_callback(event, x, y, flags, param):
    global mouse_x, mouse_y, mouse_clicked

    mouse_x, mouse_y = x, y

    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_clicked = True


def safe_eval_expression(expression: str):
    allowed_ops = {
        ast.Add: op.add,
        ast.Sub: op.sub,
        ast.Mult: op.mul,
        ast.Div: op.truediv,
        ast.Mod: op.mod,
        ast.Pow: op.pow,
        ast.USub: op.neg,
        ast.UAdd: op.pos,
    }

    def eval_node(node):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("Invalid value")

        if isinstance(node, ast.BinOp):
            return allowed_ops[type(node.op)](
                eval_node(node.left),
                eval_node(node.right),
            )

        if isinstance(node, ast.UnaryOp):
            return allowed_ops[type(node.op)](eval_node(node.operand))

        raise ValueError("Invalid expression")

    tree = ast.parse(expression, mode="eval")
    return eval_node(tree.body)


class DisplayBox:
    def __init__(self, text="0", buttons=None, height=65, gap=15, radius=18):
        self.text = text
        self.buttons = buttons
        self.height = height
        self.gap = gap
        self.radius = radius

    def draw(self, frame):
        if not self.buttons:
            return frame

        x_min = min(btn.pos[0] for btn in self.buttons)
        y_min = min(btn.pos[1] for btn in self.buttons)
        x_max = max(btn.pos[0] + btn.width for btn in self.buttons)

        x1 = x_min
        y1 = y_min - self.height - self.gap
        x2 = x_max
        y2 = y_min - self.gap

        overlay = frame.copy()

        utility.draw_rounded_rect(
            overlay,
            (x1, y1),
            (x2, y2),
            (35, 35, 35),
            radius=self.radius,
        )

        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        utility.draw_rounded_rect(
            frame,
            (x1, y1),
            (x2, y2),
            (220, 220, 220),
            radius=self.radius,
            thickness=2,
        )

        text = str(self.text) if self.text else "0"
        text = text[-18:]

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.95
        thickness = 2

        text_size = cv2.getTextSize(text, font, scale, thickness)[0]
        text_x = x2 - text_size[0] - 18
        text_y = y1 + ((y2 - y1) + text_size[1]) // 2

        cv2.putText(
            frame,
            text,
            (text_x, text_y),
            font,
            scale,
            (255, 255, 255),
            thickness,
        )

        return frame


class Button:
    def __init__(self, pos, text, size=(70, 70)):
        self.pos = pos
        self.width = size[0]
        self.height = size[1]
        self.text = text

    def is_hovered(self, x, y):
        bx, by = self.pos
        return bx <= x <= bx + self.width and by <= y <= by + self.height

    def draw(self, frame, hover=False, clicked=False):
        x, y = self.pos
        overlay = frame.copy()

        base_color = (55, 55, 55)
        hover_color = (120, 180, 255)
        clicked_color = (255, 255, 255)
        operator_color = (70, 120, 200)
        action_color = (90, 90, 90)
        equal_color = (50, 160, 100)

        if self.text in ["+", "-", "*", "/", "%", "(", ")"]:
            base_color = operator_color
        elif self.text in ["AC", "DEL"]:
            base_color = action_color
        elif self.text == "=":
            base_color = equal_color

        if clicked:
            color = clicked_color
            text_color = (0, 0, 0)
        elif hover:
            color = hover_color
            text_color = (255, 255, 255)
        else:
            color = base_color
            text_color = (255, 255, 255)

        utility.draw_rounded_rect(
            overlay,
            (x, y),
            (x + self.width, y + self.height),
            color,
            radius=16,
        )

        alpha = 0.95 if clicked else 0.75 if hover else 0.45
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        utility.draw_rounded_rect(
            frame,
            (x, y),
            (x + self.width, y + self.height),
            (220, 220, 220),
            radius=16,
            thickness=2,
        )

        font_scale = 0.65 if len(self.text) > 1 else 0.85
        thickness = 2

        text_size = cv2.getTextSize(
            self.text,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            thickness,
        )[0]

        text_x = x + (self.width - text_size[0]) // 2
        text_y = y + (self.height + text_size[1]) // 2

        cv2.putText(
            frame,
            self.text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            thickness,
        )


def handle_button_press(value):
    global calculator_text

    if value == "AC":
        calculator_text = ""

    elif value == "DEL":
        calculator_text = calculator_text[:-1]

    elif value == "=":
        try:
            if calculator_text.strip():
                result = safe_eval_expression(calculator_text)
                calculator_text = str(round(result, 6)).rstrip("0").rstrip(".")
        except Exception:
            calculator_text = "Error"

    elif value == "%":
        try:
            if calculator_text.strip():
                result = safe_eval_expression(calculator_text) / 100
                calculator_text = str(round(result, 6)).rstrip("0").rstrip(".")
        except Exception:
            calculator_text = "Error"

    else:
        if calculator_text == "Error":
            calculator_text = ""

        calculator_text += value


calc_button_values = [
    ["AC", "DEL", "%", "/"],
    ["7", "8", "9", "*"],
    ["4", "5", "6", "-"],
    ["1", "2", "3", "+"],
    ["0", ".", "(", ")"],
    ["="],
]


def create_calculator_buttons(frame):
  """
  Create calculator buttons based on the frame size and predefined button values.
  Args:
      frame (numpy.ndarray): The input video frame to determine button placement. 
  Returns:
      list: A list of Button objects with their positions and labels set according to the frame dimensions and the predefined button layout. 
  """
  
  h, w = frame.shape[:2]

  button_size = 70
  gap = 8
  rows = len(calc_button_values)
  cols = 4

  grid_width = cols * button_size + (cols - 1) * gap
  grid_height = rows * button_size + (rows - 1) * gap

  start_x = (w - grid_width) // 2
  start_y = (h - grid_height) // 2 + 40

  buttons = []

  for i, row in enumerate(calc_button_values):
      for j, value in enumerate(row):
          if value == "=":
              x = start_x
              y = start_y + i * (button_size + gap)
              width = grid_width
          else:
              x = start_x + j * (button_size + gap)
              y = start_y + i * (button_size + gap)
              width = button_size

          buttons.append(Button((x, y), value, (width, button_size)))

  return buttons


def detect_hand_pointer_and_click(frame):
    annotated_image, hand_landmarks = hand_detector.draw_landmarks(
        frame,
        to_draw_center_point=False,
        to_draw_bounding_box=False,
        to_put_handle_label=False,
    )

    if len(hand_landmarks) == 0:
        return annotated_image, None, False

    landmark_list, _, _, _ = hand_landmarks[0]

    index_finger = hand_detector.fingerTips[1]
    middle_finger = hand_detector.fingerTips[2]

    index_point = (
        landmark_list[index_finger][1],
        landmark_list[index_finger][2],
    )

    is_joined = hand_detector.is_fingers_joined_2(
        index_finger,
        middle_finger,
        annotated_image,
        landmark_list,
        threshold=0.25,
    )

    cv2.circle(
        annotated_image,
        index_point,
        10,
        (0, 255, 255) if not is_joined else (0, 255, 0),
        cv2.FILLED,
    )

    return annotated_image, index_point, is_joined


def calculator(frame):
    global mouse_x, mouse_y, mouse_clicked, calculator_text, last_click_time

    buttons = create_calculator_buttons(frame)

    display = DisplayBox(
        text=calculator_text if calculator_text else "0",
        buttons=buttons,
    )
    display.draw(frame)

    annotated_image, hand_point, hand_clicked = detect_hand_pointer_and_click(frame)

    now = time.time()

    for button in buttons:
        mouse_hover = button.is_hovered(mouse_x, mouse_y)

        hand_hover = False
        if hand_point is not None:
            hand_hover = button.is_hovered(hand_point[0], hand_point[1])

        hover = mouse_hover or hand_hover
        clicked = False

        if mouse_clicked and mouse_hover:
            if now - last_click_time > CLICK_DELAY:
                handle_button_press(button.text)
                last_click_time = now
                clicked = True

        if hand_clicked and hand_hover:
            if now - last_click_time > CLICK_DELAY:
                handle_button_press(button.text)
                last_click_time = now
                clicked = True

        button.draw(annotated_image, hover=hover, clicked=clicked)

    mouse_clicked = False

    cv2.putText(
        annotated_image,
        "Mouse click or join index+middle fingers to press",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )

    return annotated_image


if __name__ == "__main__":
    video_capture_template(
        video_source=0,
        loop_forever=False,
        custom_logic=calculator,
        window_name="Calculator",
        mouse_callback=mouse_callback,
        enable_screenshot=True
    )