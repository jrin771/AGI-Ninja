import random
import time
import threading

import pygame
import cv2
import mediapipe as mp

from fruit import Fruit, fruit_names
import calculations

# settings
FPS = 25
GAME_WIDTH = 1280 # width of pygame window (height will be auto resized to 16:9 ratio)

# global constant variables
GAME_HEIGHT = round(GAME_WIDTH * 0.5625) # 16:9 ratio
WANTED_WIDTH, WANTED_HEIGHT = 1920, 1080 # the size we expect
CAP_WIDTH, CAP_HEIGHT = 1280, 720 # size of video capture
MAX_FRUIT_HEIGHT = GAME_HEIGHT / 4 # from top of screen
ROUND_COOLDOWN = 2 # seconds
MP_POSE = mp.solutions.pose
BACKGROUND_IMAGE = "images/game_background.jpg"
transformer_meme = "images/transformer_arch.png"  
AGI_IMAGE = "images/agi-but-real.png"
AGI_IMAGE_LOADED = pygame.image.load(AGI_IMAGE)
AGI_background = pygame.transform.scale(AGI_IMAGE_LOADED, (GAME_WIDTH, GAME_HEIGHT))


# initialize all imported pygame modules
pygame.mixer.init()
pygame.font.init()

def ratio(x):
    return round(x * (GAME_WIDTH / WANTED_WIDTH))

# fonts
SMALL_NUMBER_FONT = pygame.font.SysFont('fonts/HandelGothic.ttf', ratio(30))
MAIN_NUMBER_FONT = pygame.font.Font('fonts/HandelGothic.ttf', ratio(50))
TITLE_FONT = pygame.font.Font('fonts/HandelGothic.ttf', ratio(60))

# colours
WHITE = (255, 255, 255)
GREEN = (118,185,0)

# other
KNIFE_WIDTH = ratio(10)
KNIFE_TRAIL_LIFETIME = 0.25

# create display, set window size, start clock
screen = pygame.display.set_mode([GAME_WIDTH, GAME_HEIGHT], pygame.SCALED)
pygame.display.toggle_fullscreen()
clock = pygame.time.Clock()

left_knife_trail = [] # list of tuples that store coords of knife trail, and when they're drawn
right_knife_trail = [] # right hand trail
fruits = [] # list of fruits
background_cv2_image = cv2.imread(BACKGROUND_IMAGE)

def main():
    level = 0
    total_points = 0
    explosion_alpha = 0
    start_game = False # should game start?
    running = True # is game running?
    exploding = False
    AGI = False #For now...
    
    last_round = time.time() # time that last round started
    start_fruit = None
    dead_fruit = None
    left_circles = []
    right_circles = []

    # create pose object used to motion track the pose of the user
    with MP_POSE.Pose(
        min_detection_confidence = 0.7,
        min_tracking_confidence=0.5,
        model_complexity=0) as pose:

        # open the webcamera and set the capture's resolution
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)

        # while the webcam capture session is open and the game loop is running
        while cap.isOpened() and running:
            start = time.time()
            clock.tick(FPS) # tick the clock relative to the FPS of the game
            success, frame = cap.read()

            if not success: # stopped camera?
                continue
            
            # process webcam to track pose, and draw it on image_to_display
            results, image_to_display = calculations.find_and_draw_pose(
                pose, 
                frame,
                background_cv2_image)
            
            # transform image_to_display into a pygame surface
            cam_img = calculations.array_img_to_pygame(image_to_display, GAME_WIDTH, GAME_HEIGHT)
            screen.fill(GREEN)
            screen.blit(cam_img, (0, 0))

            # generate fruit if we have none, and after intermission has ended
            if (len(fruits) == 0 and time.time() - last_round > ROUND_COOLDOWN 
                and start_game
                and start_fruit == None
                and dead_fruit is None):
                level += 1 # new level
                new_fruits = random.randrange(2, 6) # number of new fruits to generate in one round
                make_new_fruits(new_fruits, level) # give mutable reference to list of fruits that will be displayed

            # pass mutable reference of knife_trail lists, add trails based on results, and get coords of hands
            left_hand, right_hand = calculations.knife_trails_and_find_hands(
                results, 
                left_knife_trail, 
                right_knife_trail, 
                GAME_WIDTH, 
                GAME_HEIGHT)

            # delete cut fruit, see if we touched a bomb
            for fruit in fruits:
                if results.pose_landmarks and dead_fruit is None:
                    new_points, exploding = process_fruit(
                        fruit,
                        left_hand,
                        right_hand,
                        left_circles,
                        right_circles)
                    total_points += new_points
                    
                    if exploding:
                        break
            
            for fruit in fruits:
                fruit.draw(screen)
            
            # reset list containing pos of knife trail circles for cutting
            left_circles = []
            right_circles = []

            # remove old knife trail points from knife_trail lists, and draw new lines
            if len(left_knife_trail) >= 2:
                for i, point in enumerate(left_knife_trail):
                    coords, time_painted = point
                    
                    # if the knife trail point still has lifetime
                    if time_painted + KNIFE_TRAIL_LIFETIME > time.time(): 

                        # if there's another knife trail point after this one
                        if len(left_knife_trail) - 1 > i:                       # anti-aliased line
                            left_circles += calculations.knife_trail(
                                screen, 
                                GREEN, 
                                coords, 
                                left_knife_trail[i + 1][0], 
                                radius=KNIFE_WIDTH)
                    else:
                        left_knife_trail.remove(point) # remove knife trail point because its lifetime is now over

            if len(right_knife_trail) >= 2:
                for i, point in enumerate(right_knife_trail):
                    coords, time_painted = point

                    if time_painted + KNIFE_TRAIL_LIFETIME > time.time():
                        if len(right_knife_trail) - 1 > i:
                            right_circles += calculations.knife_trail(
                                screen, 
                                GREEN, 
                                coords, 
                                right_knife_trail[i + 1][0], 
                                radius=KNIFE_WIDTH)
                    else:
                        right_knife_trail.remove(point)

             # if start fruit exists (in the beginning)
            if type(start_fruit) is Fruit: 
                destroy = draw_start_end_fruit(
                    start_fruit,
                    left_circles,
                    right_circles,
                    last_round) # pass reference of start_fruit

                if destroy:

                    last_round = time.time()
                    start_fruit = None
                    start_game = True
            elif start_game == False: # no start fruit yet, but game hasn't started
                img_size = (ratio(320), ratio(400))
                fruit_pos = (
                    GAME_WIDTH / 2 - img_size[0] / 2, 
                    GAME_HEIGHT / 2 - img_size[1] / 2)
                start_fruit = Fruit( # create fruit object
                    name="Start Fruit", 
                    img_filepath="images/watermelon.png", 
                    starting_point=fruit_pos,
                    size=img_size,
                    velocity=0,
                    points=0)
                last_round = time.time()
            elif exploding or (level >= 3 and total_points < 0 and len(fruits) == 0):
                # make transformer meme if it doesn't exist already
                if dead_fruit is None:
                    img_size = (ratio(300), ratio(300))
                    fruit_pos = (
                        GAME_WIDTH / 2 - img_size[0] / 2, 
                        (GAME_HEIGHT - GAME_HEIGHT / 4) - img_size[1] / 2)
                    dead_fruit = Fruit(
                        name="transformer meme lol",
                        img_filepath= transformer_meme,
                        starting_point=fruit_pos,
                        size=img_size,
                        velocity=0,
                        points=0
                    )
                    last_round = time.time()


                # stop all fruits
                for fruit in fruits:
                    fruit.velocity = 0
                
                # fade to white (explosion)
                if explosion_alpha < 200:
                        explosion_alpha += 5

                # YOU LOST text
                explosion = pygame.Surface([GAME_WIDTH, GAME_HEIGHT], pygame.SRCALPHA, 32)
                explosion = explosion.convert_alpha()
                explosion.fill((255, 255, 255, explosion_alpha))
                screen.blit(explosion, (0, 0))

                lost_text = TITLE_FONT.render('NVIDIA Installer Failed' if exploding else 'No Compute No Funding', False, GREEN)
                screen.blit(lost_text, (
                    round(GAME_WIDTH / 2) - lost_text.get_width() / 2, 
                    round(GAME_HEIGHT / 3) - lost_text.get_height() / 2))
                
                # transformer meme
                destroy = draw_start_end_fruit(
                    dead_fruit, 
                    left_circles, 
                    right_circles, 
                    last_round)

                # start new game if transformer meme is cut
                if destroy:
                    print("Starting new game!")
                    exploding = False
                    fruits.clear()
                    level = 0
                    total_points = 0
                    last_round = time.time()
                    dead_fruit = None
                    explosion_alpha = 0

            # if no more fruits, set cooldown
            if (len(fruits) == 0 
                and time.time() - last_round > ROUND_COOLDOWN
                and level > 0
                and not dead_fruit is None):
                last_round = time.time()
                print(f"Level {level} done!")
            
            # update display, render menu
            end = time.time()
            fps = round(1 / (end - start), 1)
            display_menu(
                fps,
                level if start_game else None, 
                total_points if start_game else None, 
                len(fruits) if start_game else None)
            pygame.display.update()

            # keyboard press events
            keys_pressed = pygame.key.get_pressed()

            # escape key used to close game
            if keys_pressed[pygame.K_ESCAPE]:
                pygame.event.post(pygame.event.Event(pygame.QUIT))

            # pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit() # quit pygame window, stop game loop
                    running = False 
            
            if level == 42 or total_points > 150: 
                AGI = True 
            
            if AGI: #ok this isn't what I originally intended but it's extremely funny so I'm going to keep it because it's so cursed. 
                screen.blit(AGI_background, (0, 0)) 
                pygame.display.update() 
                


def draw_start_end_fruit(
    fruit: Fruit, 
    left_circles: list, 
    right_circles: list, 
    last_round: float) -> bool:
    circle_radius = round(fruit.get_length() * 0.65)
    hit_left = False
    
    # wait so that you don't accidentaly cut the start or end fruit just by touching it
    if time.time() - last_round >= ROUND_COOLDOWN: 
        for left_circle in left_circles:
            if calculations.distance_2D(fruit.get_centre(), left_circle) <= circle_radius:
                hit_left = True
                break

        if hit_left: # must hit with both hands!
            for right_circle in right_circles:
                if calculations.distance_2D(fruit.get_centre(), right_circle) <= circle_radius:
                    return True

    pygame.draw.circle(
        screen,
        GREEN,
        fruit.get_centre(),
        radius=circle_radius,
        width=round(fruit.get_length() / 15))
    fruit.rotation += 2
    fruit.draw(screen)
    return False

def cut_fruit(fruit: Fruit, exists_in_list=True):    
    if exists_in_list: # if fruit is in the fruits list
        fruits.remove(fruit)

    return fruit.points

def display_menu(fps, level=None, total_points=None, fruits_left=None):
    fps_text = SMALL_NUMBER_FONT.render(f'tFLOPs: {fps}', False, GREEN)
    screen.blit(fps_text, (ratio(20), GAME_HEIGHT - fps_text.get_height() - ratio(20)))

    if type(fruits_left) is int:
        fruits_left_text = SMALL_NUMBER_FONT.render(f'Fruits left: {fruits_left}', False, GREEN)
        screen.blit(
            fruits_left_text, 
                (GAME_WIDTH - fruits_left_text.get_width() - ratio(20), 
                GAME_HEIGHT - fruits_left_text.get_height() - ratio(20)))

    if type(total_points) is int:
        points_text = MAIN_NUMBER_FONT.render(f'Compute: {total_points}', False, WHITE)
        screen.blit(points_text, (ratio(20), ratio(20)))

    if type(level) is int:
        round_text = MAIN_NUMBER_FONT.render(f'Level {level}', False, WHITE)
        screen.blit(round_text, (GAME_WIDTH - round_text.get_width() - ratio(20), ratio(20)))
    else:
        starting_title = TITLE_FONT.render(f'Cut with both hands to begin!', False, GREEN)
        screen.blit(starting_title, (
            round(GAME_WIDTH / 2) - starting_title.get_width() / 2, 
            GAME_HEIGHT - starting_title.get_height() - ratio(30)))

    title_text = TITLE_FONT.render(f'AGI Ninja', False, GREEN)
    screen.blit(title_text, (round(GAME_WIDTH / 2) - title_text.get_width() / 2, ratio(20)))

def fruit_hit_circles(fruit: Fruit, points: list):
    for point in points:
        if fruit.rect.collidepoint(point): # check if it's in rectangle first
            if calculations.colliding_fruit(point, fruit):
                return True
    return False

def process_fruit(
    fruit: Fruit, 
    left_hand, right_hand, 
    left_circles,
    right_circles) -> tuple:
    points = 0
    bomb_touched = False

    # if fruit is under the screen and it's on its way down
    if fruit.y > GAME_HEIGHT and not fruit.going_up:
        fruits.remove(fruit)
        points -= fruit.points * 2
        return points, bomb_touched
        

    if fruit.name == "bomb":
        if (fruit.rect.collidepoint(left_hand) # if either hands touch the bomb
            or fruit.rect.collidepoint(right_hand)):
                bomb_touched = True
    else:
        if screen.get_rect().collidepoint((fruit.x, fruit.y)): # if left hand is in frame
            if fruit_hit_circles(fruit, left_circles): # separate if's to not check hit circles twice (laggy)
                points = cut_fruit(fruit)
                return points, bomb_touched

            if fruit_hit_circles(fruit, right_circles):
                points = cut_fruit(fruit)
                return points, bomb_touched
    
    # if reached max_fruit_height, time to go down!
    if fruit.get_centre()[1] <= MAX_FRUIT_HEIGHT and fruit.going_up:
        fruit.going_up = False
        fruit.velocity *= -1
    else:
        fruit.y -= fruit.velocity
        fruit.rotation += abs(round(fruit.velocity / 3))
    return points, bomb_touched

def make_new_fruits(number_of_fruits, level):
    probabilities = []

    # no bombs until round 5
    for fruit_n in fruit_names:
        if fruit_n[0] == "bomb" and level < 5: # name of fruit
            probabilities.append(0)
        else:
            probabilities.append(fruit_n[4]) # rarity of fruit

    random_fruits = random.choices(fruit_names, weights = probabilities, k = number_of_fruits)

    for fruit in random_fruits:
        fruit_name, img_path, points, velocity, _, img_size = fruit # unwrap list containing details of fruit
        print(f"Generating a new {fruit_name}")

        # ratio velocity
        velocity = ratio(velocity)
        
        # random velocity change
        velocity *= 1 + random.random() / 2

        random_x = random.randrange(1, GAME_WIDTH - img_size[0]) # random horizontal starting point
        random_y = random.randrange(GAME_HEIGHT, GAME_HEIGHT * 3) # random start point under screen
        
        new_fruit = Fruit( # create fruit object
            name=fruit_name, 
            img_filepath=img_path, 
            starting_point=(random_x, random_y),
            size=(ratio(img_size[0]), ratio(img_size[1])),
            velocity=velocity,
            points=points)
        fruits.append(new_fruit) # add fruit to list of fruits that will be displayed

if __name__ == '__main__':
    main()
