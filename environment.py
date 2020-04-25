import logging
import time
from multiprocessing import Lock
from multiprocessing.pool import Pool

import cv2
import mss
import numpy as np
import pytesseract
from PIL import Image
from pynput.keyboard import Controller, Key


class ScoreParser:
    def __init__(self):
        self.pool = None
        self.task = None
        self.lock = Lock()

    def __enter__(self):
        self.pool = Pool(1)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pool.close()
        self.pool.join()

    def parse_async(self, image, callback):
        with self.lock:
            if self.task is None:
                def inner_callback(value):
                    callback(value)
                    with self.lock:
                        self.task = None

                self.task = self.pool.apply_async(pytesseract.image_to_string, [image],
                                                  kwds={'config': '--oem 1 --psm 7 '  # 8
                                                                  '-c tessedit_char_whitelist=0123456789'},
                                                  callback=inner_callback)
                return True
            else:
                return False


class Environment:
    GAME_OVER = np.asarray(Image.open('game-over.png').convert('L'))
    i = 0

    def __init__(self):
        self.mss = mss.mss()
        self.keyboard = Controller()
        self.score_parser = ScoreParser()

        self.sct = None

        self.game = None
        self.last_score = None
        self.score = None

    def __enter__(self):
        self.sct = self.mss.__enter__()
        self.score_parser.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.sct = None
        self.mss.__exit__(exc_type, exc_val, exc_tb)
        self.score_parser.__exit__(exc_type, exc_val, exc_tb)

    def reset(self):
        self.unpause()

        im = self._screenshot()
        _, _, game_over = self._state(im)

        self.__exit__(None, None, None)

        self.last_score = self.score = 0
        self.game = None

        # Game over, just hit play again
        if game_over:
            self.keyboard.press(Key.space)
            time.sleep(0.1)
            self.keyboard.release(Key.space)
            time.sleep(0.5)
            self.keyboard.press(Key.space)
            time.sleep(0.1)
            self.keyboard.release(Key.space)
        # In the middle of the game, quit to menu and start play
        else:
            self.keyboard.press(Key.esc)
            time.sleep(0.1)
            self.keyboard.release(Key.esc)
            time.sleep(0.5)
            self.keyboard.press(Key.esc)
            time.sleep(0.1)
            self.keyboard.release(Key.esc)
            time.sleep(0.5)
            self.keyboard.press(Key.space)
            time.sleep(0.1)
            self.keyboard.release(Key.space)

        time.sleep(1)

        self.__enter__()

        im = self._screenshot()
        return self._state(im)

    def step(self, a_dir, a_jump, duration=0.1):
        self.unpause()

        try:
            if a_dir is not None:
                self.keyboard.press(a_dir)
            if a_jump is not None:
                self.keyboard.press(a_jump)
            time.sleep(duration)
        finally:
            if a_dir is not None:
                self.keyboard.release(a_dir)
            if a_jump is not None:
                self.keyboard.release(a_jump)

        im = self._screenshot()

        self.pause()

        return self._state(im)

    def _state(self, im):
        game_over_slice = im[175:, 21:469]
        game_over = self._is_game_over(game_over_slice)

        game = im[:445]
        state = self._preprocess_game(game)

        score = im[443:, 70:]
        score = self._parse_score(score)

        if game_over:
            score -= 100

        return state, score, game_over

    def _screenshot(self):
        if self.sct is None:
            raise ValueError("Environment not started. Did you forget to call __enter__?")

        im = np.array(self.sct.grab((140, 129, 780, 609)))  # BGR 640 x 480
        im = im[:, 75:-75]  # Remove edges
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # BGR -> RGB
        return im

    def _preprocess_game(self, game):
        ret, game = cv2.threshold(game, 50, 1, cv2.THRESH_BINARY)
        game = game.astype(np.int8)

        game_diff = game - self.game if self.game is not None else game - game
        self.game = game

        state = np.stack([game, game_diff], axis=2)
        return state

    def _parse_score(self, score):
        score = score[:, :score.shape[1] // 2]
        ret, score = cv2.threshold(score, 127, 255, cv2.THRESH_BINARY_INV)

        def callback(score: str):
            try:
                score = int(score)
            except ValueError:
                score = self.score
            logging.debug("Setting score to %d", score)
            self.score = score

        self.score_parser.parse_async(score, callback=callback)

        return self.score

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

    def pause(self):
        try:
            self.keyboard.press('p')
            time.sleep(0.05)
        finally:
            self.keyboard.release('p')

    def unpause(self):
        try:
            self.keyboard.press('q')
            time.sleep(0.05)
        finally:
            self.keyboard.release('q')


if __name__ == '__main__':
    env = Environment()
    with env:
        state, score, game_over = env.reset()
        state = Image.fromarray(state[:, :, 0].astype(np.uint8) * 255).show()
        print(f"Score: {score}, Game over: {game_over}")
