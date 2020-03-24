import unittest
import itertools

import numpy as np
from _cluster_profiler import Mixin

np.set_printoptions(suppress=True)

class TestMixin(unittest.TestCase):
    def test_angle_between_vectors(self):
        vectors_input = [
            [1, 0, 0], [0, 1, 0], [0, 0, 1],
            [-1, 0, 0], [0, -1, 0], [0, 0, -1],
            [1, 1, 0], [-1, -1, 0], [1, 1, 1]
        ]

        print(f"Generating {len(vectors_input) ** 2} input tests...")
        i = 0

        for vector_1 in vectors_input:
            for vector_2 in vectors_input:
                angle_between = Mixin.angle_between_vectors(vector_1, vector_2)
                angle_between_r = Mixin.angle_between_vectors(vector_2, vector_1)

                # Check that the angle between (a, b) is the same as (b, a)
                self.assertEqual(angle_between, angle_between_r)

                # Compute the angle expanding out passages
                dot = sum([vector_1[i] * vector_2[i] for i in range(3)])
                magnitude_1 = np.sqrt(sum([vector_1[i] ** 2 for i in range(3)]))
                magnitude_2 = np.sqrt(sum([vector_2[i] ** 2 for i in range(3)]))

                if abs(dot / (magnitude_1 * magnitude_2) - 1) < 1e-14:
                    angle_between_reference = 0.
                else:
                    angle_between_reference = np.arccos(dot / (magnitude_1 * magnitude_2)) / np.pi * 180

                # Check against reference value
                self.assertEqual(round(angle_between, 5), round(angle_between_reference, 5))
                passed = (round(angle_between, 5) == round(angle_between_reference, 5))

                # Check angle between vector and itself is zero
                if vector_1 == vector_2:
                    self.assertEqual(round(angle_between, 5), 0.)
                    self.assertEqual(round(angle_between_reference, 5), 0.)

                # Display info
                print(f"Input ({i:02d}): [{vector_1[0]:3}, {vector_1[1]:3}, {vector_1[2]:3}] | [{vector_2[0]:3},"
                      f" {vector_2[1]:3}, {vector_2[2]:3}]\t --> Angle between: {angle_between:>5.2f} \t Test "
                      f"passed: {passed}")
                i += 1

    def test_rotation_matrix_from_vectors(self):
        vectors_input = [
            [1, 0, 0], [0, 1, 0], [0, 0, 1],
            [-1, 0, 0], [0, -1, 0], [0, 0, -1],
            [1, 1, 0], [-1, -1, 0], [1, 1, 1],
            [-1, 1, 0], [1, -1, 0], [-1, -1, -1]
        ]

        z_axis_unit_vector = [0, 0, 1]
        i = 0
        print('\n')
        print(f"Generating {len(vectors_input)} input tests...")
        for vector in vectors_input:
            rot_matrix = Mixin.rotation_matrix_from_vectors(vector, z_axis_unit_vector)

            # Check that the angle between reference vec and rotated vec is zero
            vector_output = Mixin.apply_rotation_matrix(rot_matrix, vector)
            rot_angle = Mixin.angle_between_vectors(vector_output, z_axis_unit_vector)
            self.assertAlmostEqual(rot_angle, 0., places=10)

            # Check that the rotation matrix and its inverse combine to the identity matrix
            rot_matrix_r = Mixin.rotation_matrix_from_vectors(z_axis_unit_vector, vector)
            self.assertAlmostEqual(np.linalg.det(np.matmul(rot_matrix, rot_matrix_r)), 1, places=10)

            # Check that applying the inverse transformation works
            vector_output_r = Mixin.apply_rotation_matrix(rot_matrix, z_axis_unit_vector, inverse=True)
            rot_angle_r = Mixin.angle_between_vectors(vector, vector_output_r)
            self.assertAlmostEqual(rot_angle_r, 0, places=10)

            # Gather test results
            passed = (round(float(rot_angle), 10) == 0.)

            # Print info
            print(f"Input ({i:02d}): [{vector[0]:3}, {vector[1]:3}, {vector[2]:3}] \t Output after rotation:"
                  f" [{vector_output[0]:5.2f}, {vector_output[1]:5.2f}, {vector_output[2]:5.2f}]\t ---> Test "
                  f"passed: {passed}")
            i += 1

if __name__ == '__main__':
    unittest.main()


