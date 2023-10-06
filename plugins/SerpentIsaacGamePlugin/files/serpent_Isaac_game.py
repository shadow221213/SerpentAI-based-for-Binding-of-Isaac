import time

from serpent.game import Game

from .api.api import IsaacAPI

from serpent.utilities import Singleton




class SerpentIsaacGame(Game, metaclass=Singleton):
    
    def __init__( self, **kwargs ):
        kwargs["platform"] = "steam"
        
        kwargs["window_name"] = "Binding of Isaac: Repentance"
        
        kwargs["app_id"] = "250900"
        kwargs["app_args"] = None
        
        super( ).__init__(**kwargs)
        
        
        self.api_class = IsaacAPI
        self.api_instance = None
        
        self.frame_transformation_pipeline_string = "RESIZE:100x100|GRAYSCALE|FLOAT"
    
    @property
    def screen_regions( self ):
        return dict(
                MENU_NEW_GAME = (126, 322, 164, 357),
                CHARACTER_SELECT_DIFFICULTY_NORMAL = (258, 720, 282, 737),
                CHARACTER_SELECT_DIFFICULTY_HARD = (298, 716, 322, 735),
                HUD_HEALTH = (33, 121, 102, 333),
                HUD_COINS = (95, 67, 117, 117),
                HUD_BOMBS = (118, 67, 141, 117),
                HUD_KEYS = (142, 67, 167, 117),
                HUD_ITEM_ACQUIRED = (81, 167, 113, 787),
                HUD_MAP = (32, 800, 125, 910),
                HUD_MAP_CENTER = (71, 845, 85, 861),
                HUD_HEART_1 = (12, 84, 32, 106),
                HUD_HEART_2 = (12, 108, 32, 130),
                HUD_HEART_3 = (12, 132, 32, 154),
                HUD_HEART_4 = (12, 156, 32, 178),
                HUD_HEART_5 = (12, 180, 32, 202),
                HUD_HEART_6 = (12, 204, 32, 226),
                HUD_HEART_7 = (32, 84, 52, 106),
                HUD_HEART_8 = (32, 108, 52, 130),
                HUD_HEART_9 = (32, 132, 52, 154),
                HUD_HEART_10 = (32, 156, 52, 178),
                HUD_HEART_11 = (32, 180, 52, 202),
                HUD_HEART_12 = (32, 204, 52, 226),
                # HUD_BOSS_HP = (519, 371, 522, 592),
                HUD_BOSS_HP = (14, 485, 30, 713),
                HUD_BOSS_SKULL = (7, 462, 35, 492),
                GAME_ISAAC_DOOR_TOP = (43, 435, 134, 524),
                GAME_ISAAC_DOOR_RIGHT = (202, 760, 307, 825),
                GAME_ISAAC_DOOR_BOTTOM = (359, 435, 444, 524),
                GAME_ISAAC_DOOR_LEFT = (202, 138, 307, 198)
        )
    
    @property
    def ocr_presets( self ):
        presets = {
                "SAMPLE_PRESET": {
                        "extract": {
                                "gradient_size": 1,
                                "closing_size":  1
                        },
                        "perform": {
                                "scale":              10,
                                "order":              1,
                                "horizontal_closing": 1,
                                "vertical_closing":   1
                        }
                }
        }
        
        return presets
    
    def after_launch( self ):
        self.is_launched = True
        
        time.sleep(5)
        
        self.window_id = self.window_controller.locate_window(self.window_name)
        
        self.window_controller.resize_window(self.window_id, 960, 540)
        self.window_controller.move_window(self.window_id, 0, 0)
        self.window_controller.focus_window(self.window_id)
        
        self.window_geometry = self.extract_window_geometry( )
        
        print(self.window_geometry)
