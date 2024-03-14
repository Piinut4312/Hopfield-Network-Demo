import numpy as np
import random
from model import HopfieldNetwork
from dataset import Dataset, load_dataset
import pygame

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
DARK_GRAY = (130, 130, 130)
GRAY = (170, 170, 170)
LIGHT_GRAY = (210, 210, 210)

def print_data(data: np.ndarray, shape: tuple):
    w, h = shape
    for i in range(h):
        print(data[w*i:w*(i+1)])
    print()


def draw_data(surface: pygame.Surface, data: np.ndarray, colormap, shape: tuple, left: int, top: int, cell_size: int=10):
    for i in range(shape[1]):
        for j in range(shape[0]):
            rect_left = left + cell_size*j
            rect_top = top + cell_size*i
            pygame.draw.rect(
                surface, 
                colormap[data[j+i*shape[0]]], 
                pygame.Rect(rect_left, rect_top, cell_size, cell_size)
            )


class NetworkManager:

    def __init__(self, train_file_path: str, test_file_path: str, data_shape: tuple):
        self.train_dataset = load_dataset(train_file_path, data_shape)
        self.test_dataset = load_dataset(test_file_path, data_shape)
        self.data_shape = data_shape
        self.network = HopfieldNetwork(data_shape[0]*data_shape[1])
        self.network.train(self.train_dataset)


    def sample(self, source: str='train'):
        if source == 'train':
            return self.train_dataset.sample()
        elif source == 'test':
            return self.test_dataset.sample()
        raise ValueError('Unknown source name: '+source)
    

    def sample_pair(self):
        return random.choice(list(zip(self.train_dataset.data, self.test_dataset.data)))
    

def add_noise(data: np.ndarray, flip_prob: float=0.25):
    return np.multiply(data, np.random.choice([-1, 1], size=data.size, p=[flip_prob, 1-flip_prob]))
        

def draw_text(surface: pygame.surface.Surface, content: str, font: pygame.font.Font, color, pos):
    text = font.render(content, True, color)
    text_rect = text.get_rect(center=pos)
    surface.blit(text, text_rect)


class Button(pygame.sprite.Sprite):

    def __init__(self, center_pos, shape, font, text='text', command=None):
        self.center_pos = center_pos
        self.shape = shape
        self.text = text
        self.command = command
        self.color = {
            'normal': DARK_GRAY,
            'hover': GRAY,
        }
        self.state = 'normal'
        self.font = font
        self.surface = pygame.Surface(shape)
        self.render()


    def render(self):
        self.surface.fill(self.color[self.state])
        self.text_surface = self.font.render(self.text, True, BLACK)
        self.text_rect = self.text_surface.get_rect(center=(self.shape[0]/2, self.shape[1]/2))
        self.surface.blit(self.text_surface, self.text_rect)


    def is_inside(self, x, y):
        x_min = self.center_pos[0] - self.shape[0]/2
        x_max = self.center_pos[0] + self.shape[0]/2
        y_min = self.center_pos[1] - self.shape[1]/2
        y_max = self.center_pos[1] + self.shape[1]/2
        return x_min <= x <= x_max and y_min <= y <= y_max


def main():

    basic_shape = (9, 12)
    bonus_shape = (10, 10)
    data_folder = 'Hopfield_dataset/'

    basic_manager = NetworkManager(data_folder+'Basic_Training.txt', data_folder+'Basic_Testing.txt', basic_shape)
    bonus_manager = NetworkManager(data_folder+'Bonus_Training.txt', data_folder+'Bonus_Testing.txt', bonus_shape)

    screen_shape = (800, 500)
    cell_size = 20

    display_top = 40
    text_y = 310
    status_y = 350

    colormap = {1: BLACK, -1: WHITE}

    pygame.init()

    screen = pygame.display.set_mode(screen_shape)
    pygame.display.set_caption("Hopfield Network")

    large_font = pygame.font.Font(pygame.font.get_default_font(), 24)
    small_font = pygame.font.Font(pygame.font.get_default_font(), 16)

    clock = pygame.time.Clock()
    running = True

    current_manager = basic_manager
    original_data = current_manager.sample('train')
    input_data = add_noise(original_data)
    output_hist = current_manager.network.predict(input_data, 10)
    output_data = input_data.copy()
    animation_cnt = -1
    network_state = 'idle'
    program_state = 'idle'

    basic_train_button = Button((150, 400), (240, 50),  small_font, 'Sample from basic train data', None)
    basic_test_button = Button((400, 400), (240, 50),  small_font, 'Sample from basic test data', None)
    bonus_train_button = Button((150, 460), (240, 50),  small_font, 'Sample from bonus train data', None)
    bonus_test_button = Button((400, 460), (240, 50),  small_font, 'Sample from bonus test data', None)
    run_button = Button((650, 400), (240, 50),  small_font, 'Start processing', None)
    noise_button = Button((650, 460), (240, 50),  small_font, 'Add noise to input', None)

    button_list = [basic_train_button, basic_test_button, bonus_train_button, bonus_test_button, run_button, noise_button]

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                for button in button_list:
                    if not button.is_inside(event.pos[0], event.pos[1]):
                        continue
                    if 'basic' in button.text:
                        current_manager = basic_manager
                    elif 'bonus' in button.text:
                        current_manager = bonus_manager
                    elif 'noise' in button.text:
                        input_data = add_noise(input_data)
                    elif 'Start' in button.text or 'Resume' in button.text:
                        program_state = 'run'
                        network_state = 'run'
                        button.text = 'Pause processing'
                    elif 'Pause' in button.text:
                        program_state = 'idle'
                        network_state = 'pause'
                        button.text = 'Resume processing'
                    elif 'Restart' in button.text:
                        program_state = 'run'
                        network_state = 'run'
                        animation_cnt = -1
                        button.text = 'Pause processing'

                    if 'train' in button.text:
                        original_data = current_manager.sample('train')
                        input_data = np.copy(original_data)
                    elif 'test' in button.text:
                        original_data, input_data = current_manager.sample_pair()
                            
                    if 'Sample' in button.text or 'noise' in button.text:
                        network_state = 'idle'
                        output_hist = current_manager.network.predict(input_data, 10)
                        output_data = input_data.copy()
                        animation_cnt = -1
                        run_button.text = 'Start processing'


        screen.fill(LIGHT_GRAY)

        display_w = cell_size*current_manager.data_shape[0]
        gap = (screen_shape[0]-3*display_w)/4

        match program_state:
            case 'run':
                animation_cnt += 1
                if animation_cnt >= len(output_hist):
                    animation_cnt = -1
                    program_state = 'idle'
                    network_state = 'finish'
                    run_button.text = 'Restart processing'
                else:
                    output_data = output_hist[animation_cnt]       

        match network_state:
            case 'idle':
                draw_text(screen, "Not processed yet", large_font, DARK_GRAY, (3*gap+2.5*display_w, status_y))
            case 'run':
                draw_text(screen, "Processing"+"."*((animation_cnt//5)%4), large_font, DARK_GRAY, (3*gap+2.5*display_w, status_y))
            case 'pause':
                draw_text(screen, "Paused", large_font, DARK_GRAY, (3*gap+2.5*display_w, status_y))
            case 'finish':
                draw_text(screen, "Done!", large_font, (100, 255, 100), (3*gap+2.5*display_w, status_y))

        draw_data(screen, original_data, colormap, current_manager.data_shape, gap, display_top, cell_size)
        draw_data(screen, input_data, colormap, current_manager.data_shape, 2*gap+display_w, display_top, cell_size)
        draw_data(screen, output_data, colormap, current_manager.data_shape, 3*gap+2*display_w, display_top, cell_size) 

        draw_text(screen, "Ground truth", large_font, BLACK, (gap+0.5*display_w, text_y))
        draw_text(screen, "Input", large_font, BLACK, (2*gap+1.5*display_w, text_y))
        draw_text(screen, "Output", large_font, BLACK, (3*gap+2.5*display_w, text_y))

        for button in button_list:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            if button.is_inside(mouse_x, mouse_y):
                button.state = 'hover'
            else:
                button.state = 'normal'
            button.render()
            screen.blit(button.surface, button.surface.get_rect(center=button.center_pos))

        if network_state == 'run' or network_state == 'pause':
            outline_x_offset = cell_size*(animation_cnt%current_manager.data_shape[0])
            outline_y_offset = cell_size*((animation_cnt//current_manager.data_shape[0])%current_manager.data_shape[1])
            outline_left = 3*gap + 2*display_w + outline_x_offset
            outline_top = display_top + outline_y_offset
            pygame.draw.rect(screen, (255, 0, 0), pygame.Rect(outline_left, outline_top, cell_size, cell_size), 3)

        pygame.display.update()
        clock.tick(20)


if __name__ == '__main__':
    main()