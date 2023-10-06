import os.path

import matplotlib.pyplot as plt
import numpy as np

class GameData:
    
    def __init__( self ):
        self.data = dict(
                run_count = list( ),
                run_reward = list( ),
                run_boss_hp = list( ),
                run_alive_time = list( ),
                run_heart = list( )
        )
        self.model_path = "D:/SerpentAI/datasets/isaac/ppo_model/all_model"
        self.data_path = self.model_path + "/data.txt"
    
    def __adjust_annotation_positions( self, annotations, threshold = 10 ):
        positions = [a.get_position( ) for a in annotations]
        new_positions = list( )
        
        for i, pos in enumerate(positions):
            adjusted_pos = pos
            for j in range(i + 1, len(positions)):
                if np.abs(pos[0] - positions[j][0]) < threshold:
                    # Adjust the vertical position of both annotations
                    adjusted_pos = (pos[0], pos[1] + threshold)
                    break  # Adjusted position found, exit the inner loop
            
            new_positions.append(adjusted_pos)
        
        for i, annotation in enumerate(annotations):
            # if i % 2 == 0:
            #     new_positions[i] = (new_positions[i][0], new_positions[i][1] + 10)
            annotation.set_position(new_positions[i])
    
    def save_data( self, datas ):
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        
        if not os.path.exists(self.data_path):
            with open(self.data_path, "w") as file:
                data_string = "".join([f"{data} " for data in datas])
                file.write(data_string + "\n")
        else:
            with open(self.data_path, "a") as file:
                data_string = "".join([f"{data} " for data in datas])
                file.write(data_string + "\n")
    
    def load_data( self ):
        self.data["run_count"].clear( )
        self.data["run_reward"].clear( )
        self.data["run_boss_hp"].clear( )
        self.data["run_alive_time"].clear( )
        self.data["run_heart"].clear( )
        
        with open(self.data_path, "r") as file:
            for line in file:
                value = line.strip( ).split( )
                self.data["run_count"].append(int(value[0]))
                self.data["run_reward"].append(float(value[1]))
                self.data["run_boss_hp"].append(int(value[2]))
                self.data["run_alive_time"].append(float(value[3]))
                self.data["run_heart"].append(int(value[4]))
    
    def draw_data( self, path ):
        self.load_data( )
        
        run_count = self.data["run_count"]
        
        plt.figure(figsize = (16, 9))
        
        run_reward = self.data["run_reward"]
        sort_rewards_idx = np.argsort(run_reward)
        max_reward_cnt = 1
        max_rewards = list( )
        max_rewards.append([max_reward_cnt, sort_rewards_idx[-1]])
        
        for idx in reversed(sort_rewards_idx[:-1]):
            if run_reward[int(max_rewards[-1][1])] != run_reward[int(idx)]:
                max_reward_cnt += 1
                if max_reward_cnt > 3:
                    break
            
            max_rewards.append([max_reward_cnt, idx])
        
        plt.plot(run_count, run_reward, label = "Reward")
        
        rewards_annotations = list( )
        
        for cnt, max_reward in max_rewards:
            annotation = plt.annotate(
                    f'Max_{cnt}: {run_reward[max_reward]}\n'
                    f'Run: #{run_count[max_reward]}',
                    xy = (run_count[max_reward], run_reward[max_reward]),
                    xytext = (run_count[max_reward] + 5, run_reward[max_reward] + 2),
                    arrowprops = dict(facecolor = 'black', arrowstyle = '->')
            )
            rewards_annotations.append(annotation)
        
        self.__adjust_annotation_positions(rewards_annotations, 25)
        
        plt.xlabel("Round")
        plt.ylabel("Reward")
        plt.title("Run Reward")
        plt.legend( )
        plt.savefig(path + "\\run_reward_chart.jpg")
        plt.close( )
        
        plt.figure(figsize = (16, 9))
        
        run_boss_hp = self.data["run_boss_hp"]
        sort_boss_hps_idx = np.argsort(run_boss_hp)
        min_boss_hps = list( )
        min_boss_hps.append(sort_boss_hps_idx[0])
        
        for idx in sort_boss_hps_idx[1:]:
            if run_boss_hp[int(min_boss_hps[-1])] != run_boss_hp[int(idx)]:
                break
            
            min_boss_hps.append(idx)
        
        plt.plot(run_count, run_boss_hp, label = "Boss HP")
        
        boss_hps_annotations = list( )
        
        for min_boss_hp in min_boss_hps:
            annotation = plt.annotate(
                    f'Min: {run_boss_hp[min_boss_hp]}\n'
                    f'Run: #{run_count[min_boss_hp]}',
                    xy = (run_count[min_boss_hp], run_boss_hp[min_boss_hp]),
                    xytext = (run_count[min_boss_hp] + 5, run_boss_hp[min_boss_hp] + 2),
                    arrowprops = dict(facecolor = 'black', arrowstyle = '->')
            )
            boss_hps_annotations.append(annotation)
        
        self.__adjust_annotation_positions(boss_hps_annotations, 50)
        
        plt.xlabel("Round")
        plt.ylabel("Boss HP")
        plt.title("Run Boss HP")
        plt.legend( )
        plt.savefig(path + "\\run_boss_hp_chart.jpg")
        plt.close( )
        
        plt.figure(figsize = (16, 9))
        
        run_alive_time = self.data["run_alive_time"]
        sort_alive_times_idx = np.argsort(run_alive_time)
        max_alive_time_cnt = 1
        max_alive_times = list( )
        max_alive_times.append([max_alive_time_cnt, sort_alive_times_idx[-1]])
        
        for idx in reversed(sort_alive_times_idx[:-1]):
            if run_alive_time[int(max_alive_times[-1][1])] != run_alive_time[int(idx)]:
                max_alive_time_cnt += 1
                if max_alive_time_cnt > 3:
                    break
            
            max_alive_times.append([max_alive_time_cnt, idx])
        
        plt.plot(run_count, run_alive_time, label = "alive_time")
        
        alive_times_annotations = list( )
        
        for cnt, max_alive_time in max_alive_times:
            annotation = plt.annotate(
                    f'Max_{cnt}: {run_alive_time[max_alive_time]}\n'
                    f'Run: #{run_count[max_alive_time]}',
                    xy = (run_count[max_alive_time], run_alive_time[max_alive_time]),
                    xytext = (run_count[max_alive_time] + 5, run_alive_time[max_alive_time] + 2),
                    arrowprops = dict(facecolor = 'black', arrowstyle = '->')
            )
            alive_times_annotations.append(annotation)
        
        self.__adjust_annotation_positions(alive_times_annotations)
        
        plt.xlabel("Round")
        plt.ylabel("Alive Time")
        plt.title("Run Alive Time")
        plt.legend( )
        plt.savefig(path + "\\run_alive_time_chart.jpg")
        plt.close( )

        plt.figure(figsize = (16, 9))

        run_heart = self.data["run_heart"]
        sort_hearts_idx = np.argsort(run_heart)
        max_heart_cnt = 1
        max_hearts = list( )
        max_hearts.append([max_heart_cnt, sort_hearts_idx[-1]])

        for idx in reversed(sort_hearts_idx[:-1]):
            if run_heart[int(max_hearts[-1][1])] != run_heart[int(idx)]:
                max_heart_cnt += 1
                if max_heart_cnt > 3:
                    break
    
            max_hearts.append([max_heart_cnt, idx])

        plt.plot(run_count, run_heart, label = "heart")

        hearts_annotations = list( )

        for cnt, max_heart in max_hearts:
            annotation = plt.annotate(
                    f'Max_{cnt}: {run_heart[max_heart]}\n'
                    f'Run: #{run_count[max_heart]}',
                    xy = (run_count[max_heart], run_heart[max_heart]),
                    xytext = (run_count[max_heart] + 5, run_heart[max_heart]),
                    arrowprops = dict(facecolor = 'black', arrowstyle = '->')
            )
            hearts_annotations.append(annotation)

        self.__adjust_annotation_positions(hearts_annotations, 1)

        plt.xlabel("Round")
        plt.ylabel("Heart")
        plt.title("Run Heart")
        plt.legend( )
        plt.savefig(path + "\\run_heart_chart.jpg")
        plt.close( )

if __name__ == "__main__":
    game_data = GameData( )
    game_data.draw_data("D:/SerpentAI/datasets/isaac/ppo_model/all_model")