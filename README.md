# Reinforcement Learning
This repo is for playing with reinforcement learning algorithms. I am either using openai gym or ViZDoom as an environment.

I am trying several environments which includes:
- Mountain Car
    - Implemented simple Q-Learning
- Pacman
    - Implemented DQN with CNN architectures
- Breakout
    - Implemented DQN with experience replay and CNN architecture
    - Implemented DQN with GRU architecture
- Doom Game
    - Implemented double DQN on basic scenario ViZDoom

If you need help in setting up your system (mac only) with these environments, refer this:
## Openai gym
Inorder to install openai gym, run the following command in mac terminal:
- `pip install gym`

Run the below command in order to use atari environments:
- `pip install gym[atari]` 

## Setup render gym- Google Colab
Inorder to render env in colab, we need to need to install some dependencies; so run following commands in colab:
- `!apt-get install python-opengl -y`
- `!apt install xvfb -y`
- `!pip install pyvirtualdisplay`
- `!pip install piglet`
- `!apt-get install x11-utils`

After that run the following code once:
- `from pyvirtualdisplay import Display`
- `from IPython import display`
- `Display().start()`

Then put the following command, just before the episode starts (need to run only once):
- `img = plt.imshow(env.render('rgb_array'))`

Then put the following code, where we generally put `env.render()`:
- `img.set_data(env.render('rgb_array'))` # just update the data
- `display.display(plt.gcf())`
- `display.clear_output(wait=True)`

## ViZDoom
Inorder to install ViZDoom, run following commands in mac terminal (1st two commands were for installing dependencies for installing ViZDoom):
- `brew install cmake boost sdl2 wget`
- `brew cask install julia`
- `pip install vizdoom`
