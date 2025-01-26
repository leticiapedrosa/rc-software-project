
import math
import numpy as np
from rsoccer_gym.Entities import Robot
from utils.Point import Point
from utils.Geometry import Geometry


PROP_VELOCITY_MIN_FACTOR: float = 0.1
MAX_VELOCITY: float = 1.5
ANGLE_EPSILON: float = 0.1
ANGLE_KP: float = 5
MIN_DIST_TO_PROP_VELOCITY: float = 720

ADJUST_ANGLE_MIN_DIST: float = 50
M_TO_MM: float = 1000.0

K_ATTRACTIVE = 8.5  # Coeficiente da força atrativa
K_REPULSIVE = 1.0   # Coeficiente da força repulsiva
D_INFLUENCE = 0.5   # Raio de influência da força repulsiva


class Navigation:

  @staticmethod
  def degrees_to_radians(degrees):
    return degrees * (math.pi / 180.0)
  
  @staticmethod
  def radians_to_degrees(radians):
    return radians * (180.0 / math.pi)
  
  @staticmethod
  def global_to_local_velocity(vx, vy, theta):
    vx_local = vx * math.cos(theta) + vy * math.sin(theta)
    vy_local = -vx * math.sin(theta) + vy * math.cos(theta)
    return Point(vx_local, vy_local)

  @staticmethod
  def map_value(value, lLower, lHigher, rLower, rHigher):
    if (lHigher - lLower) == 0:
      return
    
    return ((value - lLower) * (rHigher - rLower) / (lHigher - lLower) + rLower)

  @staticmethod
  def goToPoint(robot: Robot, target: Point):
    target = Point(target.x * M_TO_MM, target.y * M_TO_MM)
    robot_position = Point(robot.x * M_TO_MM, robot.y * M_TO_MM)
    robot_angle = Navigation.degrees_to_radians(Geometry.normalize_angle(robot.theta, 0, 180))

    max_velocity = MAX_VELOCITY
    distance_to_target = robot_position.dist_to(target)
    kp = ANGLE_KP

    # Use proportional speed to decelerate when getting close to desired target
    proportional_velocity_factor = PROP_VELOCITY_MIN_FACTOR
    min_proportional_distance = MIN_DIST_TO_PROP_VELOCITY

    if distance_to_target <= min_proportional_distance:
      max_velocity = max_velocity * Navigation.map_value(distance_to_target, 0.0, min_proportional_distance, proportional_velocity_factor, 1.0)

    target_angle = (target - robot_position).angle()
    d_theta = Geometry.smallest_angle_diff(target_angle, robot_angle)

    if distance_to_target > ADJUST_ANGLE_MIN_DIST:
      v_angle = Geometry.abs_smallest_angle_diff(math.pi - ANGLE_EPSILON, d_theta)

      v_proportional = v_angle * (max_velocity / (math.pi - ANGLE_EPSILON))
      global_final_velocity = Geometry.from_polar(v_proportional, target_angle)
      target_velocity = Navigation.global_to_local_velocity(global_final_velocity.x, global_final_velocity.y, robot_angle)

      return target_velocity, -kp * d_theta
    else:
      return Point(0.0, 0.0), -kp * d_theta

  @staticmethod
  def calculate_attractive_force(position, goal):
      """
      Calcula a força atrativa em direção ao objetivo.
      """
      force = K_ATTRACTIVE * (np.array(goal) - np.array(position))
      return Point(force[0], force[1])

  @staticmethod
  def calculate_repulsive_force(position, obstacles):
      """
      Calcula a força repulsiva com base nos obstáculos.
      """
      total_force = np.array([0.0, 0.0])
      position = np.array(position)

      for obs in obstacles:
          obs = np.array(obs)
          distance = np.linalg.norm(position - obs)

          if distance < D_INFLUENCE and distance > 0:
              repulsive_magnitude = K_REPULSIVE * ((1 / distance) - (1 / D_INFLUENCE)) / (distance ** 2)
              direction = (position - obs) / distance
              total_force += repulsive_magnitude * direction

      return Point(total_force[0], total_force[1])

  @staticmethod
  def calculate_total_force(position, goal, obstacles):
      """
      Calcula a força total combinando forças atrativas e repulsivas.
      """
      attractive_force = Navigation.calculate_attractive_force(position, goal)
      repulsive_force = Navigation.calculate_repulsive_force(position, obstacles)
      return Point(attractive_force.x + repulsive_force.x, attractive_force.y + repulsive_force.y)
  
  @staticmethod
  def calculateAngularVelocity(robot: Robot, target: Point) -> float:
    """
    Calcula a velocidade angular necessária para alinhar o robô ao objetivo.

    :param robot: Objeto do robô atual contendo sua posição e ângulo (theta).
    :param target: Ponto de destino ao qual o robô deve se alinhar.
    :return: Velocidade angular (float).
    """
    # Coordenadas do robô e do objetivo
    robot_position = Point(robot.x, robot.y)
    target_direction = (target - robot_position).angle()  # Ângulo em direção ao objetivo
    robot_angle = Navigation.degrees_to_radians(robot.theta)  # Ângulo do robô em radianos

    # Diferença angular
    angle_diff = Geometry.smallest_angle_diff(target_direction, robot_angle)

    # Proporcionalidade para ajustar a velocidade angular
    angular_velocity = ANGLE_KP * angle_diff

    # Limitar a velocidade angular máxima (opcional, pode ajustar conforme necessário)
    max_angular_velocity = 5.0  # Valor arbitrário, ajustar conforme o modelo do robô
    angular_velocity = max(-max_angular_velocity, min(max_angular_velocity, angular_velocity))

    return angular_velocity