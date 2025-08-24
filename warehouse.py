import pygame
import math
import random
import heapq
from typing import List, Tuple, Optional, Dict
from enum import Enum
from dataclasses import dataclass

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 1400
SCREEN_HEIGHT = 900
FPS = 60
GRID_SIZE = 30
NUM_ROBOTS = 8

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (50, 150, 255)
RED = (255, 100, 100)
GREEN = (100, 255, 100)
GRAY = (128, 128, 128)
YELLOW = (255, 255, 0)
PURPLE = (200, 100, 255)
ORANGE = (255, 165, 0)
DARK_GRAY = (64, 64, 64)
LIGHT_GRAY = (192, 192, 192)

class TaskType(Enum):
    PICK = "pick"
    DELIVER = "deliver"
    IDLE = "idle"

class RobotState(Enum):
    IDLE = "idle"
    MOVING_TO_PICK = "moving_to_pick"
    PICKING = "picking"
    MOVING_TO_DELIVER = "moving_to_deliver"
    DELIVERING = "delivering"
    BLOCKED = "blocked"

@dataclass
class Task:
    id: int
    pick_location: Tuple[int, int]
    deliver_location: Tuple[int, int]
    priority: int = 1
    assigned_robot: Optional[int] = None
    completed: bool = False

@dataclass
class GridNode:
    x: int
    y: int
    walkable: bool = True
    occupied_by: Optional[int] = None  # Robot ID
    is_shelf: bool = False
    is_station: bool = False

class Vector2:
    def __init__(self, x: float = 0, y: float = 0):
        self.x = x
        self.y = y
    
    def __add__(self, other):
        return Vector2(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        return Vector2(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar):
        return Vector2(self.x * scalar, self.y * scalar)
    
    def magnitude(self):
        return math.sqrt(self.x ** 2 + self.y ** 2)
    
    def normalize(self):
        mag = self.magnitude()
        if mag > 0:
            return Vector2(self.x / mag, self.y / mag)
        return Vector2(0, 0)

class PathPlanner:
    @staticmethod
    def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])  # Manhattan distance
    
    @staticmethod
    def get_neighbors(pos: Tuple[int, int], grid_width: int, grid_height: int) -> List[Tuple[int, int]]:
        x, y = pos
        neighbors = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # 4-directional
            nx, ny = x + dx, y + dy
            if 0 <= nx < grid_width and 0 <= ny < grid_height:
                neighbors.append((nx, ny))
        return neighbors
    
    @staticmethod
    def a_star(start: Tuple[int, int], goal: Tuple[int, int], grid: List[List[GridNode]], 
               ignore_robot_id: Optional[int] = None) -> List[Tuple[int, int]]:
        if start == goal:
            return [start]
        
        grid_width = len(grid[0])
        grid_height = len(grid)
        
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: PathPlanner.heuristic(start, goal)}
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            
            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]
            
            for neighbor in PathPlanner.get_neighbors(current, grid_width, grid_height):
                nx, ny = neighbor
                node = grid[ny][nx]
                
                # Skip unwalkable nodes or nodes occupied by other robots
                if not node.walkable or (node.occupied_by and node.occupied_by != ignore_robot_id):
                    continue
                
                tentative_g_score = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + PathPlanner.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return []  # No path found

class Robot:
    def __init__(self, robot_id: int, start_x: int, start_y: int):
        self.id = robot_id
        self.grid_pos = (start_x, start_y)
        self.world_pos = Vector2(start_x * GRID_SIZE + GRID_SIZE//2, 
                                start_y * GRID_SIZE + GRID_SIZE//2)
        self.target_world_pos = Vector2(self.world_pos.x, self.world_pos.y)
        
        self.state = RobotState.IDLE
        self.current_task: Optional[Task] = None
        self.path: List[Tuple[int, int]] = []
        self.path_index = 0
        
        self.speed = 2.0
        self.carrying_item = False
        
        # Timing
        self.action_timer = 0
        self.pick_time = 30  # frames
        self.deliver_time = 30  # frames
    
    def assign_task(self, task: Task):
        self.current_task = task
        self.state = RobotState.MOVING_TO_PICK
        task.assigned_robot = self.id
    
    def update(self, grid: List[List[GridNode]], task_manager):
        self.action_timer -= 1
        
        # Move towards target position
        self._move_towards_target()
        
        # State machine
        if self.state == RobotState.IDLE:
            pass  # Wait for task assignment
            
        elif self.state == RobotState.MOVING_TO_PICK:
            if self.current_task:
                if not self.path or self.path[-1] != self.current_task.pick_location:
                    # Plan path to pick location
                    self.path = PathPlanner.a_star(
                        self.grid_pos, self.current_task.pick_location, grid, self.id
                    )
                    self.path_index = 0
                
                if self._follow_path(grid):
                    # Reached pick location
                    self.state = RobotState.PICKING
                    self.action_timer = self.pick_time
        
        elif self.state == RobotState.PICKING:
            if self.action_timer <= 0:
                self.carrying_item = True
                self.state = RobotState.MOVING_TO_DELIVER
        
        elif self.state == RobotState.MOVING_TO_DELIVER:
            if self.current_task:
                if not self.path or self.path[-1] != self.current_task.deliver_location:
                    # Plan path to delivery location
                    self.path = PathPlanner.a_star(
                        self.grid_pos, self.current_task.deliver_location, grid, self.id
                    )
                    self.path_index = 0
                
                if self._follow_path(grid):
                    # Reached delivery location
                    self.state = RobotState.DELIVERING
                    self.action_timer = self.deliver_time
        
        elif self.state == RobotState.DELIVERING:
            if self.action_timer <= 0:
                self.carrying_item = False
                if self.current_task:
                    self.current_task.completed = True
                    task_manager.complete_task(self.current_task.id)
                self.current_task = None
                self.state = RobotState.IDLE
        
        # Update grid occupation
        self._update_grid_occupation(grid)
    
    def _move_towards_target(self):
        diff = self.target_world_pos - self.world_pos
        if diff.magnitude() > self.speed:
            direction = diff.normalize()
            self.world_pos = self.world_pos + direction * self.speed
        else:
            self.world_pos = Vector2(self.target_world_pos.x, self.target_world_pos.y)
    
    def _follow_path(self, grid: List[List[GridNode]]) -> bool:
        if not self.path or self.path_index >= len(self.path):
            return True
        
        target_grid = self.path[self.path_index]
        target_world = Vector2(target_grid[0] * GRID_SIZE + GRID_SIZE//2,
                              target_grid[1] * GRID_SIZE + GRID_SIZE//2)
        
        # Check if we've reached current waypoint
        if (self.world_pos - target_world).magnitude() < 5:
            self.grid_pos = target_grid
            self.path_index += 1
            
            if self.path_index < len(self.path):
                next_grid = self.path[self.path_index]
                self.target_world_pos = Vector2(
                    next_grid[0] * GRID_SIZE + GRID_SIZE//2,
                    next_grid[1] * GRID_SIZE + GRID_SIZE//2
                )
            else:
                return True  # Reached end of path
        else:
            self.target_world_pos = target_world
        
        return False
    
    def _update_grid_occupation(self, grid: List[List[GridNode]]):
        # Clear previous occupation
        for row in grid:
            for node in row:
                if node.occupied_by == self.id:
                    node.occupied_by = None
        
        # Set current occupation
        gx, gy = self.grid_pos
        if 0 <= gx < len(grid[0]) and 0 <= gy < len(grid):
            grid[gy][gx].occupied_by = self.id
    
    def draw(self, screen):
        # Draw robot body
        color = BLUE
        if self.state == RobotState.PICKING or self.state == RobotState.DELIVERING:
            color = YELLOW
        elif self.carrying_item:
            color = GREEN
        
        pygame.draw.circle(screen, color, 
                         (int(self.world_pos.x), int(self.world_pos.y)), 12)
        pygame.draw.circle(screen, WHITE, 
                         (int(self.world_pos.x), int(self.world_pos.y)), 12, 2)
        
        # Draw robot ID
        font = pygame.font.Font(None, 20)
        text = font.render(str(self.id), True, WHITE)
        screen.blit(text, (self.world_pos.x - 5, self.world_pos.y - 7))
        
        # Draw path
        if len(self.path) > 1:
            path_points = []
            for gx, gy in self.path:
                wx = gx * GRID_SIZE + GRID_SIZE // 2
                wy = gy * GRID_SIZE + GRID_SIZE // 2
                path_points.append((wx, wy))
            
            if len(path_points) > 1:
                pygame.draw.lines(screen, color, False, path_points, 2)

class TaskManager:
    def __init__(self):
        self.tasks: Dict[int, Task] = {}
        self.next_task_id = 1
        self.completed_tasks = 0
    
    def create_task(self, pick_pos: Tuple[int, int], deliver_pos: Tuple[int, int], priority: int = 1) -> Task:
        task = Task(
            id=self.next_task_id,
            pick_location=pick_pos,
            deliver_location=deliver_pos,
            priority=priority
        )
        self.tasks[self.next_task_id] = task
        self.next_task_id += 1
        return task
    
    def get_available_tasks(self) -> List[Task]:
        return [task for task in self.tasks.values() 
                if not task.assigned_robot and not task.completed]
    
    def complete_task(self, task_id: int):
        if task_id in self.tasks:
            del self.tasks[task_id]
            self.completed_tasks += 1
    
    def assign_tasks(self, robots: List[Robot], grid: List[List[GridNode]]):
        available_tasks = self.get_available_tasks()
        idle_robots = [r for r in robots if r.state == RobotState.IDLE]
        
        # Sort tasks by priority
        available_tasks.sort(key=lambda t: t.priority, reverse=True)
        
        for task in available_tasks:
            if not idle_robots:
                break
            
            # Find closest idle robot
            best_robot = None
            best_distance = float('inf')
            
            for robot in idle_robots:
                distance = PathPlanner.heuristic(robot.grid_pos, task.pick_location)
                if distance < best_distance:
                    best_distance = distance
                    best_robot = robot
            
            if best_robot:
                best_robot.assign_task(task)
                idle_robots.remove(best_robot)

class WarehouseSimulator:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Warehouse Fleet Management System")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        
        # Grid setup
        self.grid_width = SCREEN_WIDTH // GRID_SIZE
        self.grid_height = SCREEN_HEIGHT // GRID_SIZE
        
        # Initialize grid
        self.grid = [[GridNode(x, y) for x in range(self.grid_width)] 
                     for y in range(self.grid_height)]
        self._setup_warehouse_layout()
        
        # Initialize robots
        self.robots = []
        start_positions = [(1, 1), (2, 1), (3, 1), (1, 2), (2, 2), (3, 2), (1, 3), (2, 3)]
        for i in range(min(NUM_ROBOTS, len(start_positions))):
            x, y = start_positions[i]
            self.robots.append(Robot(i, x, y))
        
        # Task management
        self.task_manager = TaskManager()
        
        # UI state
        self.paused = False
        self.show_info = True
        self.show_grid = True
    
    def _setup_warehouse_layout(self):
        # Create warehouse shelving layout
        # Shelves (obstacles)
        shelf_positions = [
            # Vertical shelving aisles
            [(8, y) for y in range(5, 15)],
            [(15, y) for y in range(5, 15)],
            [(22, y) for y in range(5, 15)],
            [(29, y) for y in range(5, 15)],
            [(36, y) for y in range(5, 15)],
            
            # Horizontal shelving
            [(x, 18) for x in range(8, 40)],
            [(x, 22) for x in range(8, 40)],
        ]
        
        for shelf_row in shelf_positions:
            for x, y in shelf_row:
                if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
                    self.grid[y][x].walkable = False
                    self.grid[y][x].is_shelf = True
        
        # Mark packing stations (delivery points)
        station_positions = [(2, 25), (4, 25), (6, 25)]
        for x, y in station_positions:
            if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
                self.grid[y][x].is_station = True
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    self.reset_simulation()
                elif event.key == pygame.K_i:
                    self.show_info = not self.show_info
                elif event.key == pygame.K_g:
                    self.show_grid = not self.show_grid
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click - create pick task
                    grid_x = event.pos[0] // GRID_SIZE
                    grid_y = event.pos[1] // GRID_SIZE
                    self._create_random_task_from_click(grid_x, grid_y)
                elif event.button == 3:  # Right click - toggle obstacle
                    grid_x = event.pos[0] // GRID_SIZE  
                    grid_y = event.pos[1] // GRID_SIZE
                    self._toggle_obstacle(grid_x, grid_y)
        
        return True
    
    def _create_random_task_from_click(self, click_x: int, click_y: int):
        # Pick location near click
        pick_pos = (click_x, click_y)
        
        # Random delivery station
        stations = [(2, 25), (4, 25), (6, 25)]
        deliver_pos = random.choice(stations)
        
        self.task_manager.create_task(pick_pos, deliver_pos)
    
    def _toggle_obstacle(self, x: int, y: int):
        if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
            node = self.grid[y][x]
            if not node.is_shelf and not node.is_station:
                node.walkable = not node.walkable
    
    def reset_simulation(self):
        # Reset robots
        start_positions = [(1, 1), (2, 1), (3, 1), (1, 2), (2, 2), (3, 2), (1, 3), (2, 3)]
        for i, robot in enumerate(self.robots):
            if i < len(start_positions):
                x, y = start_positions[i]
                robot.grid_pos = (x, y)
                robot.world_pos = Vector2(x * GRID_SIZE + GRID_SIZE//2, 
                                        y * GRID_SIZE + GRID_SIZE//2)
                robot.target_world_pos = Vector2(robot.world_pos.x, robot.world_pos.y)
                robot.state = RobotState.IDLE
                robot.current_task = None
                robot.path = []
                robot.carrying_item = False
        
        # Clear tasks
        self.task_manager.tasks.clear()
        self.task_manager.next_task_id = 1
        self.task_manager.completed_tasks = 0
    
    def update(self):
        if not self.paused:
            # Assign tasks to idle robots
            self.task_manager.assign_tasks(self.robots, self.grid)
            
            # Update all robots
            for robot in self.robots:
                robot.update(self.grid, self.task_manager)
    
    def draw(self):
        self.screen.fill(BLACK)
        
        # Draw grid
        if self.show_grid:
            for x in range(0, SCREEN_WIDTH, GRID_SIZE):
                pygame.draw.line(self.screen, DARK_GRAY, (x, 0), (x, SCREEN_HEIGHT))
            for y in range(0, SCREEN_HEIGHT, GRID_SIZE):
                pygame.draw.line(self.screen, DARK_GRAY, (0, y), (SCREEN_WIDTH, y))
        
        # Draw warehouse elements
        for y, row in enumerate(self.grid):
            for x, node in enumerate(row):
                rect = pygame.Rect(x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE)
                
                if node.is_shelf:
                    pygame.draw.rect(self.screen, GRAY, rect)
                elif node.is_station:
                    pygame.draw.rect(self.screen, GREEN, rect)
                elif not node.walkable:
                    pygame.draw.rect(self.screen, RED, rect)
        
        # Draw tasks
        for task in self.task_manager.tasks.values():
            if not task.assigned_robot:
                # Unassigned pick location
                px, py = task.pick_location
                pygame.draw.circle(self.screen, ORANGE, 
                                 (px * GRID_SIZE + GRID_SIZE//2, 
                                  py * GRID_SIZE + GRID_SIZE//2), 8)
        
        # Draw robots
        for robot in self.robots:
            robot.draw(self.screen)
        
        # Draw UI
        if self.show_info:
            self.draw_ui()
        
        pygame.display.flip()
    
    def draw_ui(self):
        info_texts = [
            "Warehouse Fleet Management System",
            f"Robots: {len(self.robots)} | Tasks: {len(self.task_manager.tasks)} | Completed: {self.task_manager.completed_tasks}",
            f"FPS: {int(self.clock.get_fps())}",
            "",
            "Controls:",
            "Left Click: Create Pick Task",
            "Right Click: Toggle Obstacle", 
            "SPACE: Pause/Resume",
            "R: Reset Simulation",
            "G: Toggle Grid",
            "I: Toggle Info",
            "",
            "Legend:",
            "Blue: Idle Robot | Yellow: Working | Green: Carrying Item",
            "Gray: Shelves | Green Squares: Packing Stations",
            "Orange: Unassigned Tasks"
        ]
        
        if self.paused:
            info_texts.insert(3, "*** PAUSED ***")
        
        y_offset = 10
        for text in info_texts:
            if text:
                color = YELLOW if "PAUSED" in text else WHITE
                surface = self.font.render(text, True, color)
                self.screen.blit(surface, (10, y_offset))
            y_offset += 20
    
    def run(self):
        running = True
        while running:
            running = self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(FPS)
        
        pygame.quit()

if __name__ == "__main__":
    simulator = WarehouseSimulator()
    simulator.run()
    