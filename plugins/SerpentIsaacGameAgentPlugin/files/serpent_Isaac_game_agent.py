import collections
import itertools
import os
import pickle
import shlex
import subprocess
import time

import numpy as np
import pyperclip
import serpent.cv
import skimage
from serpent.config import config
from serpent.frame_grabber import FrameGrabber
from serpent.game_agent import GameAgent
from serpent.input_controller import KeyboardKey

from .helpers.frame_processing import frame_to_boss_health, frame_to_hearts
from .helpers.game_data import GameData
from .helpers.ppo import SerpentPPO
from .helpers.terminal_printer import TerminalPrinter

class SerpentIsaacGameAgent(GameAgent):
    
    def __init__( self, **kwargs ):
        super( ).__init__(**kwargs)
        
        self.frame_handlers["PLAY"] = self.handle_play
        
        self.frame_handler_setups["PLAY"] = self.setup_play
        
        self.frame_handler_pause_callbacks["PLAY"] = self.handle_play_pause
        
        self.printer = TerminalPrinter( )
        
        self.draw_game_data = GameData( )
        
        self.total_path = "D:/SerpentAI"
        self.path_metadata = os.path.join(os.getcwd( ), "datasets", "isaac", "ppo_model", "all_model")
        
        self.paused_at = None
    
    @property
    def bosses( self ):
        return {
                "MONSTRO": "1010"
        }
    
    def relaunch( self ):
        self.printer.flush( )
        
        self.printer.add("")
        self.printer.add("The game appears to have crashed...")
        self.printer.add("")
        
        self.printer.add("Hot-swapping the game window the agent is looking at...")
        self.printer.add("The experiment will resume once the new game window is ready!")
        self.printer.add("")
        
        self.printer.flush( )
        
        self.game.stop_frame_grabber( )
        
        time.sleep(1)
        
        self.input_controller.tap_key(KeyboardKey.KEY_ENTER, force = True)
        
        time.sleep(1)
        
        subprocess.call(shlex.split("serpent launch Binding"))
        self.game.launch(dry_run = True)
        
        self.game.start_frame_grabber( )
        self.game.redis_client.delete((config["frame_grabber"]["redis_key"]))
        
        while self.game.redis_client.llen(config["frame_grabber"]["redis_key"]) == 0:
            time.sleep(0.1)
        
        self.game.window_controller.focus_window(self.game.window_id)
        
        time.sleep(3)
        
        self.input_controller.tap_key(KeyboardKey.KEY_ENTER)
        time.sleep(0.2)
        self.input_controller.tap_key(KeyboardKey.KEY_ENTER)
        time.sleep(0.2)
        self.input_controller.tap_key(KeyboardKey.KEY_ENTER)
        time.sleep(0.2)
        self.input_controller.tap_key(KeyboardKey.KEY_ENTER)
        time.sleep(3)
        
        self.printer.flush( )
    
    def setup_play( self ):
        self.first_run = True

        self.boss_skull_image = None
        
        self.first_hearts = None
        self.first_health = None
        
        self.passed_boss = False
        
        move_inputs = {
                "MOVE UP":         [KeyboardKey.KEY_W],
                "MOVE LEFT":       [KeyboardKey.KEY_A],
                "MOVE DOWN":       [KeyboardKey.KEY_S],
                "MOVE RIGHT":      [KeyboardKey.KEY_D],
                "MOVE TOP-LEFT":   [KeyboardKey.KEY_W, KeyboardKey.KEY_A],
                "MOVE TOP-RIGHT":  [KeyboardKey.KEY_W, KeyboardKey.KEY_D],
                "MOVE DOWN-LEFT":  [KeyboardKey.KEY_S, KeyboardKey.KEY_A],
                "MOVE DOWN-RIGHT": [KeyboardKey.KEY_S, KeyboardKey.KEY_D],
                "DON'T MOVE":      []
        }
        
        shoot_inputs = {
                "SHOOT UP":    [KeyboardKey.KEY_UP],
                "SHOOT LEFT":  [KeyboardKey.KEY_LEFT],
                "SHOOT DOWN":  [KeyboardKey.KEY_DOWN],
                "SHOOT RIGHT": [KeyboardKey.KEY_RIGHT],
                "DON'T SHOOT": []
        }
        
        self.game_inputs = dict( )
        
        for move_label, shoot_label in itertools.product(move_inputs, shoot_inputs):
            label = f"{move_label.ljust(20)}{shoot_label}"
            self.game_inputs[label] = move_inputs[move_label] + shoot_inputs[shoot_label]
        
        self.run_count = 0
        
        self.observation_count = 0
        
        self.performed_inputs = collections.deque(list( ), maxlen = 8)
        
        self.reward_10 = collections.deque(list( ), maxlen = 10)
        self.reward_100 = collections.deque(list( ), maxlen = 100)
        self.reward_1000 = collections.deque(list( ), maxlen = 1000)
        
        self.average_reward_10 = 0
        self.average_reward_100 = 0
        self.average_reward_1000 = 0
        
        self.top_reward = 0
        self.top_reward_run = 0
        
        self.run_alive_time = 0
        
        self.alive_time_10 = collections.deque(list( ), maxlen = 10)
        self.alive_time_100 = collections.deque(list( ), maxlen = 100)
        self.alive_time_1000 = collections.deque(list( ), maxlen = 1000)
        
        self.average_alive_time_10 = 0
        self.average_alive_time_100 = 0
        self.average_alive_time_1000 = 0
        
        self.top_alive_time = 0
        self.top_alive_time_run = 0
        
        self.run_boss_hp = 0
        self.run_heart = 0
        
        self.boss_hp_10 = collections.deque(list( ), maxlen = 10)
        self.boss_hp_100 = collections.deque(list( ), maxlen = 100)
        self.boss_hp_1000 = collections.deque(list( ), maxlen = 1000)
        
        self.average_boss_hp_10 = 0
        self.average_boss_hp_100 = 0
        self.average_boss_hp_1000 = 0
        
        self.best_boss_hp = 218
        self.best_boss_hp_run = 0
        
        self.death_check = False
        self.pass_check = False
        
        self.frame_buffer = None
        
        self.ppo_agent = SerpentPPO(
                frame_shape = (100, 100, 4),
                game_inputs = self.game_inputs
        )
        
        try:
            self.path_metadata = os.path.join(os.getcwd( ), "datasets", "isaac", "ppo_model", "all_model")
            self.ppo_agent.agent.restore_model(directory = self.path_metadata)
            self.restore_metadata(self.path_metadata)
        except Exception:
            pass
        
        game_frame_buffer = FrameGrabber.get_frames([0, 2, 4, 6], frame_type = "PIPELINE")
        self.ppo_agent.generate_action(game_frame_buffer)
        
        self.run_reward = 0
        
        self.multiplier_alive = 1.0
        self.multiplier_damage = 1.0
        
        self.episode_started_at = None
        
        self.shoot = None
    
    def handle_play( self, game_frame ):
        self.paused_at = None

        if self.first_run:
            self.run_count += 1

            self.__goto_boss(boss_key = self.bosses["MONSTRO"], items = ["c330", "c92", "c92", "c92"])

            self.first_run = False

            self.episode_started_at = time.time( )

            return None

        self.printer.add("")
        self.printer.add("Serpent.AI Lab - Binding of Isaac: Repentance")
        self.printer.add("Reinforcement Learning: Training a PPO Agent")
        self.printer.add("")
        self.printer.add(f"Current Run: #{self.run_count}")
        self.printer.add("")

        hearts = frame_to_hearts(game_frame, self.game)
        heart_count = len(hearts) - hearts.count(None)

        boss_health = frame_to_boss_health(game_frame, self.game)
        health_count = len(boss_health) - boss_health.count(None) + 1

        if not len(hearts):
            self.input_controller.tap_key(KeyboardKey.KEY_R, duration = 1.5)
            self.__goto_boss(boss_key = self.bosses["MONSTRO"], items = ["c330", "c92", "c92", "c92"])

            return None

        if self.first_hearts is None:
            if heart_count == 0:
                self.__restart( )

                return None
            else:
                self.first_hearts = heart_count
                self.first_health = health_count

                self.health = collections.deque(np.full((16,), self.first_hearts), maxlen = 16)
                self.boss_health = collections.deque(np.full((24,), self.first_health + 1), maxlen = 24)

        else:
            if self.__is_boss_dead(game_frame):
                health_count -= 1
                
                if self.run_reward < 36:
                    self.__restart( )

                    return None
                elif not self.pass_check:
                    self.pass_check = True

                    self.health.appendleft(heart_count)
                    self.boss_health.appendleft(health_count)

                    self.printer.flush( )
                    return None
                else:
                    self.passed_boss = True
            else:
                self.pass_check = False

        self.health.appendleft(heart_count)
        self.boss_health.appendleft(health_count)

        reward, is_alive = self.reward_isaac([None, None, game_frame, None])

        if self.frame_buffer is not None:
            self.run_reward += reward

            self.observation_count += 1

            self.analytics_client.track(event_key = "RUN_REWARD", data = dict(reward = reward))

            episode_duration = time.time( ) - self.episode_started_at
            episode_over = episode_duration >= 800

            if episode_over:
                is_alive = False

            if self.ppo_agent.agent.batch_count == 2047:
                self.printer.flush( )
                self.printer.add("Updating Isaac Model With New Data...")
                self.printer.flush( )

                self.input_controller.tap_key(KeyboardKey.KEY_ESCAPE)
                self.ppo_agent.observe(reward, terminal = (not is_alive or self.passed_boss))
                self.input_controller.tap_key(KeyboardKey.KEY_ESCAPE)

                self.frame_buffer = None

                if not episode_over:
                    time.sleep(1)
                    return None
            else:
                self.ppo_agent.observe(reward, terminal = (not is_alive or self.passed_boss))

        self.printer.add(f"Observation Count: {self.observation_count}")
        self.printer.add(f"Current Batch Size: {self.ppo_agent.agent.batch_count}")
        self.printer.add("")
        self.printer.add(f"Run Reward: {round(self.run_reward, 2)}")
        self.printer.add("")

        if is_alive and not self.passed_boss:
            self.death_check = False

            self.printer.add(f"Survival Multiplier: {round(self.multiplier_alive, 2)}")
            self.printer.add(f"Boss Damage: {round(self.multiplier_damage, 2)}")
            self.printer.add(f"Alive Time: {round(time.time( ) - self.episode_started_at, 2)}")
            self.printer.add("")
            self.printer.add(f"Average Reward (Last 10 Runs): {round(self.average_reward_10, 2)}")
            self.printer.add(f"Average Reward (Last 100 Runs): {round(self.average_reward_100, 2)}")
            self.printer.add(f"Average Reward (Last 1000 Runs): {round(self.average_reward_1000, 2)}")
            self.printer.add("")
            self.printer.add(f"Top Run Reward: {round(self.top_reward, 2)} (Run #{self.top_reward_run})")
            self.printer.add("")
            self.printer.add(f"Average Alive Time (Last 10 Runs): {round(self.average_alive_time_10, 2)}")
            self.printer.add(f"Average Alive Time (Last 100 Runs): {round(self.average_alive_time_100, 2)}")
            self.printer.add(f"Average Alive Time (Last 1000 Runs): {round(self.average_alive_time_1000, 2)}")
            self.printer.add("")
            self.printer.add(f"Top Alive Time: {self.top_alive_time} (Run #{self.top_alive_time_run})")
            self.printer.add("")
            self.printer.add(f"Average Boss HP (Last 10 Runs): {round(self.average_boss_hp_10, 2)}")
            self.printer.add(f"Average Boss HP (Last 100 Runs): {round(self.average_boss_hp_100, 2)}")
            self.printer.add(f"Average Boss HP (Last 1000 Runs): {round(self.average_boss_hp_1000, 2)}")
            self.printer.add("")
            self.printer.add(f"Best Boss HP: {self.best_boss_hp} (Run #{self.best_boss_hp_run})")
            self.printer.add("")
            self.printer.add("Latest Inputs:")
            self.printer.add("")

            for performed_input in self.performed_inputs:
                self.printer.add(performed_input)

            self.printer.flush( )

            self.frame_buffer = FrameGrabber.get_frames([0, 2, 4, 6], frame_type = "PIPELINE")
            action, label, game_input = self.ppo_agent.generate_action(self.frame_buffer)

            self.performed_inputs.appendleft(label)
            self.input_controller.handle_keys(game_input)
        else:
            self.input_controller.handle_keys([])

            if not self.death_check:
                self.death_check = True

                self.printer.flush( )
                return None
            else:
                self.analytics_client.track(event_key = "RUN_END", data = dict(run = self.run_count))

                self.printer.flush( )

                self.reward_10.appendleft(self.run_reward)
                self.reward_100.appendleft(self.run_reward)
                self.reward_1000.appendleft(self.run_reward)

                self.average_reward_10 = float(np.mean(self.reward_10))
                self.average_reward_100 = float(np.mean(self.reward_100))
                self.average_reward_1000 = float(np.mean(self.reward_1000))

                if self.run_reward > self.top_reward:
                    self.top_reward = self.run_reward
                    self.top_reward_run = self.run_count

                self.analytics_client.track(event_key = "EPISODE_REWARD",
                                            data = dict(total_reward = self.run_reward))

                self.run_alive_time = time.time( ) - self.episode_started_at

                self.alive_time_10.appendleft(self.run_alive_time)
                self.alive_time_100.appendleft(self.run_alive_time)
                self.alive_time_1000.appendleft(self.run_alive_time)

                self.average_alive_time_10 = float(np.mean(self.alive_time_10))
                self.average_alive_time_100 = float(np.mean(self.alive_time_100))
                self.average_alive_time_1000 = float(np.mean(self.alive_time_1000))

                if self.run_alive_time > self.top_alive_time:
                    self.top_alive_time = self.run_alive_time
                    self.top_alive_time_run = self.run_count

                self.run_boss_hp = max([self.boss_health[i] for i in range(3)])

                self.boss_hp_10.appendleft(self.run_boss_hp)
                self.boss_hp_100.appendleft(self.run_boss_hp)
                self.boss_hp_1000.appendleft(self.run_boss_hp)

                self.average_boss_hp_10 = float(np.mean(self.boss_hp_10))
                self.average_boss_hp_100 = float(np.mean(self.boss_hp_100))
                self.average_boss_hp_1000 = float(np.mean(self.boss_hp_1000))

                if self.run_boss_hp < self.best_boss_hp:
                    self.best_boss_hp = self.run_boss_hp
                    self.best_boss_hp_run = self.run_count

                self.run_heart = max([self.health[i] for i in range(3)])

                # if not self.run_count % 1:
                if not self.run_count % 10:
                    path_model = os.path.join(os.getcwd( ), "datasets", "isaac", "ppo_model",
                                              "model_" + str(self.run_count))
                    if not os.path.exists(path_model):
                        os.makedirs(path_model)
                    self.ppo_agent.agent.save_model(directory = path_model + "\\model",
                                                    append_timestep = False)
                    self.ppo_agent.agent.save_model(directory = self.path_metadata + "\\model",
                                                    append_timestep = False)
                    self.dump_metadata(path_model)
                    self.dump_metadata(self.path_metadata)

                self.draw_game_data.save_data([self.run_count, round(self.run_reward, 2), self.run_boss_hp,
                                               round(self.run_alive_time, 2), self.run_heart])
                self.draw_game_data.draw_data(self.path_metadata)

                self.__clear( )

                self.input_controller.tap_key(KeyboardKey.KEY_R, duration = 1.5)
                self.__goto_boss(boss_key = self.bosses["MONSTRO"], items = ["c330", "c92", "c92", "c92"])

                self.run_count += 1

                self.pass_check = False

                self.boss_skull_image = None

                self.episode_started_at = time.time( )
    
    def handle_play_pause( self ):
        if self.paused_at is None:
            self.paused_at = time.time( )
        
        # work after 15 seconds
        if time.time( ) - self.paused_at >= 15:
            self.relaunch( )
            self.first_run = True
    
    def reward_isaac( self, frames, **kwargs ):
        # Punish: Loss HP, Loss Hit
        # Reward: Staying alive, Hit boss
        
        boss_damaged_recently = len(set(self.boss_health)) > 1
        self.multiplier_damage = 1 / len(set(self.health))
        
        if boss_damaged_recently:
            self.multiplier_alive = 1.0
        elif self.multiplier_alive - 0.03 >= 0.2:
            self.multiplier_alive -= 0.03
        else:
            self.multiplier_alive = 0.2
        
        reward = 0
        is_alive = self.health[0] + self.health[1]
        
        if is_alive:
            reward += 0.5 * self.multiplier_alive
            
            if self.health[0] < self.health[1]:
                factor = self.health[1] - self.health[0]
                reward -= factor * 0.25
                
                return reward, is_alive
        
        if self.boss_health[0] < self.boss_health[1]:
            reward += (self.boss_health[1] - self.boss_health[0]) / 2 * 0.5 * self.multiplier_damage
        
        return reward, is_alive
    
    def __log( self, base, x ):
        log_base_x = np.log(x) / np.log(base)
        return log_base_x
    
    def dump_metadata( self, path ):
        metadata = dict(
                run_count = self.run_count,
                observation_count = self.observation_count,
                reward_10 = self.reward_10,
                reward_100 = self.reward_100,
                reward_1000 = self.reward_1000,
                average_reward_10 = self.average_reward_10,
                average_reward_100 = self.average_reward_100,
                average_reward_1000 = self.average_reward_1000,
                top_reward = self.top_reward,
                top_reward_run = self.top_reward_run,
                alive_time_10 = self.alive_time_10,
                alive_time_100 = self.alive_time_100,
                alive_time_1000 = self.alive_time_1000,
                average_alive_time_10 = self.average_alive_time_10,
                average_alive_time_100 = self.average_alive_time_100,
                average_alive_time_1000 = self.average_alive_time_1000,
                top_alive_time = self.top_alive_time,
                top_alive_time_run = self.top_alive_time_run,
                boss_hp_10 = self.boss_hp_10,
                boss_hp_100 = self.boss_hp_100,
                boss_hp_1000 = self.boss_hp_1000,
                average_boss_hp_10 = self.average_boss_hp_10,
                average_boss_hp_100 = self.average_boss_hp_100,
                average_boss_hp_1000 = self.average_boss_hp_1000,
                best_boss_hp = self.best_boss_hp,
                best_boss_hp_run = self.best_boss_hp_run
        )
        
        with open(path + "\\metadata.json", "wb") as f:
            f.write(pickle.dumps(metadata))
    
    def restore_metadata( self, path ):
        with open(path + "\\metadata.json", "rb") as f:
            metadata = pickle.loads(f.read( ))
        
        self.run_count = metadata["run_count"]
        self.observation_count = metadata["observation_count"]
        # self.reward_10 = metadata["reward_10"]
        # self.reward_100 = metadata["reward_100"]
        # self.reward_1000 = metadata["reward_1000"]
        # self.average_reward_10 = metadata["average_reward_10"]
        # self.average_reward_100 = metadata["average_reward_100"]
        # self.average_reward_1000 = metadata["average_reward_1000"]
        # self.top_reward = metadata["top_reward"]
        # self.top_reward_run = metadata["top_reward_run"]
        
        # self.alive_time_10 = metadata["alive_time_10"]
        # self.alive_time_100 = metadata["alive_time_100"]
        # self.alive_time_1000 = metadata["alive_time_1000"]
        # self.average_alive_time_10 = metadata["average_alive_time_10"]
        # self.average_alive_time_100 = metadata["average_alive_time_100"]
        # self.average_alive_time_1000 = metadata["average_alive_time_1000"]
        # self.top_alive_time = metadata["top_alive_time"]
        # self.top_alive_time_run = metadata["top_alive_time_run"]
        
        self.model_path = "D:/SerpentAI/datasets/isaac/ppo_model/all_model"
        self.data_path = self.model_path + "/data.txt"
        
        with open(self.data_path, "r") as file:
            for line in file:
                value = line.strip( ).split( )
                
                self.run_reward = float(value[1])
                
                self.reward_10.appendleft(self.run_reward)
                self.reward_100.appendleft(self.run_reward)
                self.reward_1000.appendleft(self.run_reward)
                
                self.average_reward_10 = float(np.mean(self.reward_10))
                self.average_reward_100 = float(np.mean(self.reward_100))
                self.average_reward_1000 = float(np.mean(self.reward_1000))
                
                if self.run_reward > self.top_reward:
                    self.top_reward = self.run_reward
                    self.top_reward_run = int(value[0])
                
                self.run_alive_time = float(value[3])
                
                self.alive_time_10.appendleft(self.run_alive_time)
                self.alive_time_100.appendleft(self.run_alive_time)
                self.alive_time_1000.appendleft(self.run_alive_time)
                
                self.average_alive_time_10 = float(np.mean(self.alive_time_10))
                self.average_alive_time_100 = float(np.mean(self.alive_time_100))
                self.average_alive_time_1000 = float(np.mean(self.alive_time_1000))
                
                if self.run_alive_time > self.top_alive_time:
                    self.top_alive_time = self.run_alive_time
                    self.top_alive_time_run = int(value[0])
                
                self.run_boss_hp = int(value[2])
                
                self.boss_hp_10.appendleft(self.run_boss_hp)
                self.boss_hp_100.appendleft(self.run_boss_hp)
                self.boss_hp_1000.appendleft(self.run_boss_hp)
                
                self.average_boss_hp_10 = float(np.mean(self.boss_hp_10))
                self.average_boss_hp_100 = float(np.mean(self.boss_hp_100))
                self.average_boss_hp_1000 = float(np.mean(self.boss_hp_1000))
                
                if self.run_boss_hp < self.best_boss_hp:
                    self.best_boss_hp = self.run_boss_hp
                    self.best_boss_hp_run = int(value[0])
        
        # self.boss_hp_10 = metadata["boss_hp_10"]
        # self.boss_hp_100 = metadata["boss_hp_100"]
        # self.boss_hp_1000 = metadata["boss_hp_1000"]
        # self.average_boss_hp_10 = metadata["average_boss_hp_10"]
        # self.average_boss_hp_100 = metadata["average_boss_hp_100"]
        # self.average_boss_hp_1000 = metadata["average_boss_hp_1000"]
        # self.best_boss_hp = metadata["best_boss_hp"]
        # self.best_boss_hp_run = metadata["best_boss_hp_run"]
    
    def __goto_boss( self, boss_key, items = None ):
        self.input_controller.tap_key(KeyboardKey.KEY_SPACE)
        time.sleep(1)
        self.input_controller.tap_key(KeyboardKey.KEY_GRAVE)
        time.sleep(0.5)
        
        if items is not None:
            for item in items:
                pyperclip.copy(f"giveitem {item}")
                self.input_controller.tap_keys([KeyboardKey.KEY_LEFT_CTRL, KeyboardKey.KEY_V], duration = 0.1)
                time.sleep(0.1)
                self.input_controller.tap_key(KeyboardKey.KEY_ENTER)
                time.sleep(0.1)
        
        pyperclip.copy(f"goto s.boss.{boss_key}")
        self.input_controller.tap_keys([KeyboardKey.KEY_LEFT_CTRL, KeyboardKey.KEY_V], duration = 0.1)
        time.sleep(0.1)
        
        self.input_controller.tap_key(KeyboardKey.KEY_ENTER)
        time.sleep(0.1)
        self.input_controller.tap_key(KeyboardKey.KEY_ENTER)
        time.sleep(0.5)
        self.input_controller.tap_key(KeyboardKey.KEY_ENTER)
        time.sleep(0.5)
        self.input_controller.tap_key(KeyboardKey.KEY_ENTER)
        time.sleep(0.2)
        
        self.analytics_client.track(event_key = "RUN_START", data = dict(run = self.run_count))
    
    def __is_boss_dead( self, game_frame ):
        gray_boss_skull = serpent.cv.extract_region_from_image(
                game_frame.grayscale_frame,
                self.game.screen_regions["HUD_BOSS_SKULL"]
        )
        
        if self.boss_skull_image is None:
            self.boss_skull_image = gray_boss_skull
        
        is_dead = False
        
        if skimage.measure.compare_ssim(gray_boss_skull, self.boss_skull_image) < 0.5:
            is_dead = True
        
        return is_dead
    
    def __clear( self ):
        self.health = collections.deque(np.full((16,), 24), maxlen = 16)
        self.boss_health = collections.deque(np.full((24,), 218), maxlen = 24)
        
        self.performed_inputs.clear( )
        
        self.frame_buffer = None
        
        self.first_hearts = None
        self.first_health = None
        
        self.passed_boss = False
        
        self.run_reward = 0
    
    def __restart( self ):
        self.__clear( )
        
        self.input_controller.tap_key(KeyboardKey.KEY_R, duration = 1.5)
        self.__goto_boss(boss_key = self.bosses["MONSTRO"], items = ["c330", "c92", "c92", "c92"])
        
        self.episode_started_at = time.time( )