# Experiment: Manipulating the Latent Space of a World Model to Change Game Physics

I wanted to explore the latent space and the things one can achieve without further training. I explored the latent space to achieve two things: 1) increase the ball speed and 2) change the bricks design with time.

Why this experiment is cool and important at the same time? If this is possible it means we can completely interpret the latent space and programmatically manipulate it according to our requirement without further or with minimal training. I conducted this experiment to tweak the game physics using latent manipulation or minimal fine tuning. 

### checking whether the world model can reconstruct frames with faster ball movement.  


### Visualise the the latent space and visually differentiate between latent spaces of faster and slower ball speed.

I generated gameplay videos of the game "Breakout" with different ball speeds using a Random agent. The videos were then passed through the world model to obtain the latent representations. Upon visualizing the latent space, I observed that the latent representations of the faster ball speed gameplay were distinct from those of the slower ball speed gameplay. This suggests that the world model captures the dynamics of the game, including the speed of the ball, in its latent space.

<!-- ![Latent Space Visualization](outputs/breakout_fast.gif) -->
<p float="left">
  <img src="outputs/breakout_fast.gif" width="200" />
  <img src="outputs/breakout_slow.gif" width="200" />
</p>

Since the ball passes faster through certain areas, I noticed that certain region in the latent space has more frequent updates compared to others. This indicates that the world model is sensitive to the speed of the ball and can differentiate between faster and slower gameplay. The question is how to find out the exact latent vector that represents the ball speed and how to manipulate it to achieve faster ball movement in the game.

Methods to increase the ball speed:
- Latent interpolation 
  Compute and store the latent representations of both the faster and slower gameplay.  Compute the average difference between them to identify the latent vector that represents the velocity of the ball. Add this velocity vector to any latent vector to reconstruct frames with faster ball speed. 

  Either consider manipulating the entire latent or only the visually identified area representing the ball movement. 

- Fine tuning