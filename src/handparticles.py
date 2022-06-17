import cv2
import time
import numpy as np
import handtracking as ht
import particlesystem as ps


### Parameters ###
color_palette = [(68, 106, 70), (130, 162, 132), (228, 174, 197), (255, 196, 221)]
number_of_particles = 50
min_particle_radius = 10
max_particle_radius = 50
min_initial_particle_speed = 3
max_initial_particle_speed = 5
max_particle_speed = 20
collision_radius_threshold = 5
collision_speed_threshold = 3
bounce = 1.1
particle_speed_decay_factor = 0.98

adjust_resolution = False
resolution_factor_x = 2
resolution_factor_y = 2

draw_hands = False
### ---------- ###


cap = cv2.VideoCapture(0)
if adjust_resolution:
    success, img = cap.read()
    new_width = img.shape[1] * resolution_factor_x
    new_height = img.shape[0] * resolution_factor_y
    cap = cv2.VideoCapture(0)
    cap.set(3, new_width)
    cap.set(4, new_height)
success, img = cap.read()

detector = ht.HandDetector()
particle_system = ps.ParticleSystem(number_of_particles,
                                    img.shape, 
                                    min_particle_radius, 
                                    max_particle_radius,
                                    color_palette, 
                                    min_initial_particle_speed,
                                    max_initial_particle_speed, 
                                    max_particle_speed,
                                    particle_speed_decay_factor,
                                    collision_radius_threshold,
                                    collision_speed_threshold,
                                    bounce)


previous_time = 0
last_right_index_tip_pos = np.array([0, 0])

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.find_hands(img, draw=draw_hands)
    lm_list_right, lm_list_left = detector.find_landmark_positions(img)
    
    if detector.right_left[0] and len(lm_list_right) > 0:
        right_index_tip_pos = np.array(
            lm_list_right[detector.mp_hands.HandLandmark.INDEX_FINGER_TIP][1:]
        )
        hand_speed = right_index_tip_pos - last_right_index_tip_pos
        last_right_index_tip_pos = right_index_tip_pos
        particle_system.update(
            right_index_tip_pos, 
            hand_speed, 
            use_collider=True
        ) 
    else:
        particle_system.update(use_collider=False)

    for particle in particle_system.particles:
        cv2.circle(
            img, 
            particle.pos.astype(int), 
            particle.radius, 
            particle.color, 
            cv2.FILLED
        )

    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time

    cv2.putText(
        img, 
        str(int(fps)), 
        (10, 20), 
        cv2.FONT_HERSHEY_PLAIN, 
        1, 
        (255, 255, 255), 
        1
    )
    cv2.imshow("Bubbles", img)
    cv2.waitKey(1)
