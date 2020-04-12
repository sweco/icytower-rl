import logging
import time

import cv2
import mss
import numpy as np
import pytesseract
from PIL import Image
from pynput.keyboard import Controller, Key


class Environment:
    GAME_OVER = np.asarray(Image.open('game-over.png').convert('L'))
    i = 0

    def __init__(self):
        self.mss = mss.mss()
        self.keyboard = Controller()
        self.sct = None

        self.game, self.score, self.action = None, None, None
        self.reset(reset_using_keyboard=False)

    def __enter__(self):
        self.sct = self.mss.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.sct = None
        self.mss.__exit__(exc_type, exc_val, exc_tb)

        if self.action is not None:
            self.keyboard.release(self.action)

    def reset(self, reset_using_keyboard=True):
        self.score = 0
        self.action = None
        self.game = None

        if reset_using_keyboard:
            self.keyboard.press(Key.space)
            time.sleep(0.1)
            self.keyboard.release(Key.space)
            time.sleep(0.5)
            self.keyboard.press(Key.space)
            time.sleep(0.1)
            self.keyboard.release(Key.space)
            time.sleep(1)

    def state(self):
        if self.sct is None:
            raise ValueError("Environment not started. Did you forget to call __enter__?")

        im = np.array(self.sct.grab((140, 129, 780, 609)))  # BGR 640 x 480
        im = im[:, 75:-75]  # Remove edges
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # BGR -> RGB

        game_over_slice = im[175:, 21:469]
        game_over = self._is_game_over(game_over_slice)

        game = im[:445]
        state = self._preprocess_game(game)

        score = im[443:, 70:]
        score = self._parse_score(score)

        return state, score, game_over

    def _preprocess_game(self, game):
        ret, game = cv2.threshold(game, 50, 1, cv2.THRESH_BINARY)
        game = game.astype(np.int8)

        game_diff = game - self.game if self.game is not None else game
        self.game = game

        state = np.concatenate([game, game_diff])
        return state

    def _parse_score(self, score) -> str:
        score = score[:, :score.shape[1] // 2]
        ret, score = cv2.threshold(score, 127, 255, cv2.THRESH_BINARY_INV)
        score = pytesseract.image_to_string(score, config='--oem 1 --psm 7 '  # 8
                                                          '-c tessedit_char_whitelist=0123456789')

        try:
            score = int(score)
            if score < self.score:
                score = self.score
            self.score = score
        except ValueError:
            score = self.score

        return score

    def _is_game_over(self, game_over_slice):
        middle_row_idx = self.GAME_OVER.shape[0] // 2
        middle_row = self.GAME_OVER[middle_row_idx]

        on_idxs = middle_row > 0
        middle_row_conf = np.mean(game_over_slice[:, on_idxs] == middle_row[on_idxs], axis=1)
        middle_row_pos = np.argmax(middle_row_conf)
        middle_row_conf = middle_row_conf[middle_row_pos]
        start_pos = middle_row_pos - middle_row_idx

        logging.debug("Start pos: %d, Conf: %f", start_pos, middle_row_conf)

        if start_pos >= 0 and start_pos + self.GAME_OVER.shape[0] <= game_over_slice.shape[0] and middle_row_conf > 0.9:
            game_over = game_over_slice[start_pos:start_pos + self.GAME_OVER.shape[0]]

            on_idxs = self.GAME_OVER > 0
            equal_pixels = game_over[on_idxs] == self.GAME_OVER[on_idxs]
            conf = np.mean(equal_pixels)
            logging.debug(f" Game over: %f", conf)
            return conf > 0.9
        else:
            return False

    def act(self, action):
        if self.action is not None:
            self.keyboard.release(self.action)

        self.action = action
        self.keyboard.press(action)


if __name__ == '__main__':
    with Environment() as env:
        game, score, game_over = env.state()
        game = Image.fromarray(game).show()
        print(f"Score: '{score}', Game over: {game_over}")
