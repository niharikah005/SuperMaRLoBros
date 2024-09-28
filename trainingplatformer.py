def train_agent():
    env = DummyVecEnv([lambda: Pls_learn()])  

   
    model_params = {
        "policy": "CnnPolicy",  
        "env": env,             
        "learning_rate": 0.0003,  
        "n_steps": 2048,        
        "batch_size": 64,       
        "n_epochs": 10,         
        "gamma": 0.99,          
        "gae_lambda": 0.95,     
        "clip_range": 0.2,      
        "verbose": 1,          
        "tensorboard_log": "./ppo_tensorboard" 
    }

    model = PPO(**model_params)
    model.learn(total_timesteps=10000)
    model.save("ppo_platformer")

    obs = env.reset()
    for _ in range(1000):
        action, _ = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()

    env.close()

if __name__ == "__main__":
    train_agent()





