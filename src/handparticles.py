import time
from argparse import ArgumentParser, Namespace

import cv2
import numpy as np

import handtracking as ht
import particlesystem as ps

COLORS = [
    (68, 106, 70),
    (130, 162, 132),
    (228, 174, 197),
    (255, 196, 221),
]


def main(args: Namespace):
    cap = cv2.VideoCapture(0)

    if args.resolution_factor != 1:
        success, img = cap.read()
        cap = cv2.VideoCapture(0)
        cap.set(3, img.shape[1] * args.resolution_factor)
        cap.set(4, img.shape[0] * args.resolution_factor)

    success, img = cap.read()

    detector = ht.HandDetector(
        model_path=args.model_path,
    )
    particle_system = ps.ParticleSystem(
        args.n_particles,
        img.shape,
        args.p_min_radius,
        args.p_max_radius,
        COLORS,
        args.p_min_start_speed,
        args.p_max_start_speed,
        args.p_max_speed,
        args.p_speed_decay_factor,
        args.collision_radius_threshold,
        args.collision_speed_threshold,
        args.bounce,
    )

    previous_time = 0
    last_right_index_tip_pos = np.array([0, 0])

    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        detection_result = detector.detect(img)
        if detection_result is not None:
            landmarks = detector.get_landmark_positions(
                detection_result=detection_result
            )
            transformed_landmarks = (
                detector.transform_landmark_positions_to_image(
                    landmarks=landmarks, img=img
                )
            )

            if len(transformed_landmarks) > 0:
                right_index_tip_pos = np.array(
                    transformed_landmarks[
                        detector.hand_landmark.INDEX_FINGER_TIP
                    ][1:]
                )
                hand_speed = right_index_tip_pos - last_right_index_tip_pos
                last_right_index_tip_pos = right_index_tip_pos
                particle_system.update(
                    right_index_tip_pos, hand_speed, use_collider=True
                )
            else:
                particle_system.update()

        for particle in particle_system.particles:
            cv2.circle(
                img,
                particle.pos.astype(int),
                particle.radius,
                particle.color,
                cv2.FILLED,
            )

        if args.draw_hand:
            img = detector.draw_landmarks(
                img=img, detection_result=detection_result
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
            1,
        )
        cv2.imshow("Bubbles", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    parser = ArgumentParser("Handparticles")
    detector_group = parser.add_argument_group("Detector")
    detector_group.add_argument(
        "--model_path", type=str, help="Path to the MediaPipe model"
    )
    detector_group.add_argument(
        "--draw_hand",
        action="store_true",
        help="Draw the landmarks of the interacting hand",
    )
    particles_group = parser.add_argument_group("Particle System")
    particles_group.add_argument(
        "--n_particles", type=int, help="Number of particles"
    )
    particles_group.add_argument(
        "--p_min_radius", type=int, help="The minimum radius of a particle"
    )
    particles_group.add_argument(
        "--p_max_radius", type=int, help="The maximum radius of a particle"
    )
    particles_group.add_argument(
        "--p_min_start_speed",
        type=int,
        help="The minimum initial speed of a particle",
    )
    particles_group.add_argument(
        "--p_max_start_speed",
        type=int,
        help="The maximum initial speed of a particle",
    )
    particles_group.add_argument(
        "--p_max_speed",
        type=float,
        help="The highest possible speed a particle can have per dimension",
    )
    particles_group.add_argument(
        "--p_speed_decay_factor",
        type=float,
        help="Particle speed multiplier per timestep",
    )
    particles_group.add_argument(
        "--collision_radius_threshold",
        type=float,
        help="Distance between particle and collider to trigger a collision",
    )
    particles_group.add_argument(
        "--collision_speed_threshold",
        type=float,
        help="Minimum collider speed for a collision",
    )
    particles_group.add_argument(
        "--bounce",
        type=float,
        help="Bounce factor to make collisions more pronounced",
    )

    resolution_group = parser.add_argument_group("Resolution")
    resolution_group.add_argument(
        "--resolution_factor",
        type=float,
        default=1,
        help="Image shape multiplier. Defaults to 1",
    )
    args = parser.parse_args()
    main(args)
