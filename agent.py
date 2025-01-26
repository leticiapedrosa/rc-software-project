from utils.ssl.Navigation import Navigation
from utils.ssl.base_agent import BaseAgent
from utils.Point import Point

class ExampleAgent(BaseAgent):
    def __init__(self, id=0, yellow=False):
        super().__init__(id, yellow)

    #alterar a função abaixo
    def decision(self):
        if len(self.targets) == 0:
            return

        # Posição atual do robô
        robot_position = Point(self.robot.x, self.robot.y)

        # Posição do objetivo (primeiro da lista de alvos)
        goal = self.targets[0]

        # Converter obstáculos para uma lista de pontos
        obstacles = [Point(robot.x, robot.y) for robot in self.obstacles.values()]

        # Calcular força total do campo potencial
        total_force = Navigation.calculate_total_force(robot_position, goal, obstacles)

        # Limitar velocidade para o máximo permitido
        target_velocity_global = Point(total_force[0], total_force[1])
        target_velocity = Navigation.global_to_local_velocity(
            target_velocity_global.x, target_velocity_global.y, self.robot.theta
        )

        # Calcular velocidade angular para alinhar o robô ao objetivo
        target_angle_velocity = Navigation.calculateAngularVelocity(self.robot, goal)

        self.set_vel(target_velocity)
        self.set_angle_vel(target_angle_velocity)

        return

    def post_decision(self):
        pass
