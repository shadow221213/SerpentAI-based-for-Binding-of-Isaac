import serpent.cv
from serpent.visual_debugger.visual_debugger import VisualDebugger

visual_debugger = VisualDebugger( )

HEART_COLORS = {
        "lefts":  {
                (232, 0, 0):    "RED",
                (97, 117, 163): "SOUL"
                # (63, 63, 63):   "BLACK"
        },
        "rights": {
                (232, 0, 0):     "RED",
                (97, 117, 163):  "SOUL",
                (72, 87, 121):   "SOUL"
                # (63, 63, 63):    "BLACK"
                # (255, 255, 255): "ETERNAL"
        }
}

def frame_to_hearts( game_frame, game ):
    heart_positions = range(1, 13)
    heart_labels = [f"HUD_HEART_{position}" for position in heart_positions]
    
    hearts = list( )
    
    for heart_label in heart_labels:
        heart = serpent.cv.extract_region_from_image(game_frame.frame, game.screen_regions[
            heart_label])
        
        left_heart_pixel = tuple(heart[3, 5, :])
        right_heart_pixel = tuple(heart[3, 17, :])
        unknown_heart_pixel = tuple(heart[3, 11, :])
        # print(heart_label, left_heart_pixel, right_heart_pixel, unknown_heart_pixel)
        
        if unknown_heart_pixel == (255, 255, 255):
            hearts.append("HOLY")
        
        hearts.append(HEART_COLORS["lefts"].get(left_heart_pixel))
        hearts.append(HEART_COLORS["rights"].get(right_heart_pixel))
    
    return hearts

BOSS_HEALTH_COLORS = {
        (212, 0, 0): "RED",
        (138, 0, 0): "RED"
}

def frame_to_boss_health( game_frame, game ):
    health_label = "HUD_BOSS_HP"
    
    hearts = list( )
    
    health = serpent.cv.extract_region_from_image(game_frame.frame, game.screen_regions[health_label])
    
    health_pixels = health[15, :, :]
    # print(health_pixels)
    
    for health_pixel in health_pixels:
        health_pixel_tuple = tuple(health_pixel)
        hearts.append(BOSS_HEALTH_COLORS.get(health_pixel_tuple))
    
    return hearts