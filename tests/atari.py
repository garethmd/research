import dqn


def test_stack_should_be_correct_shape():
    env = TorchEnv(gym.make(environment))
    stack = env.reset()
    assert stack.shape == (4, 210, 160, 3) # 4 frames, 210x160, 3 channels

def test_stack_should_be_correct_dtype():
    env = TorchEnv(gym.make(environment))
    stack = env.reset()
    assert stack.dtype == np.uint8

def test_step_should_be_correct_dtype():
    env = TorchEnv(gym.make(environment))
    _ = env.reset()
    stack, r, terminated = env.step(0)
    assert stack.dtype == np.uint8


test_stack_should_be_correct_shape()
test_stack_should_be_correct_dtype()
test_step_should_be_correct_dtype()

def test_step_returns_correct_shape():
    env = TorchEnv(gym.make(environment), transforms=pipeline)
    s = env.reset()
    a = env.sample()
    s_prime, r, terminated = env.step(a)
    s_prime = torch.cat([s, s_prime])[-4:]
    assert s_prime.shape == (4, 1, 84, 84), f"s_prime should be of shape (4, 1, 84, 84), but is {s_prime.shape}"

test_step_returns_correct_shape()

def test_step_different_state():
    env = TorchEnv(gym.make(environment), transforms=pipeline)
    s = env.reset()
    a = env.sample()
    s_prime, r, terminated = env.step(a)
    s_prime = torch.cat([s, s_prime])[-4:]
    assert not torch.all(torch.eq(s, s_prime)), "s and s_prime should be different"

test_step_different_state()

def test_should_return_correct_number_of_samples():
    env = TorchEnv(gym.make(environment), transforms=pipeline)
    replay_memory = ExperienceReplay(replay_memory_size)
    fill(replay_memory, env, fill_size=10)
    samples = replay_memory.sample(3)
    assert len(samples) == 3

test_should_return_correct_number_of_samples()
def test_sample_should_contain_5_elements():
    env = TorchEnv(gym.make(environment), transforms=pipeline)
    replay_memory = ExperienceReplay(replay_memory_size)
    fill(replay_memory, env, fill_size=10)
    samples = replay_memory.sample(3)
    assert len(samples[0]) == 5 # s, a, r, s_prime, terminated

test_sample_should_contain_5_elements()

def test_dqn_should_return_correct_s_shape_given_batch():
    batch = replay_memory.sample(3)
    s_j, a_j, r_j, s_prime_j, terminated_j = list(zip(*batch))
    s_j = torch.stack(s_j).squeeze(2)
    assert dqn(s_j).shape == torch.Size([3, 4])

test_dqn_should_return_correct_s_shape_given_batch()
def test_action_shape_should_be_correct_give_atari_collate():    
    batch = replay_memory.sample(3)
    _, a_j, r_j, s_prime_j, terminated_j = atari_collate(batch)
    assert a_j.shape == torch.Size([3])

test_action_shape_should_be_correct_give_atari_collate()

def test_reward_shape_should_be_correct_give_atari_collate():
    batch = replay_memory.sample(3)
    _, a_j, r_j, s_prime_j, terminated_j = atari_collate(batch)
    assert r_j.shape == torch.Size([3])

test_reward_shape_should_be_correct_give_atari_collate()

def test_s_shape_should_be_correct_give_atari_collate():
    batch = replay_memory.sample(3)
    s, _, _, _, _ = atari_collate(batch)
    assert s.shape == torch.Size([3, 4, 84, 84])

test_s_shape_should_be_correct_give_atari_collate()

def test_s_prime_shape_should_be_correct_give_atari_collate():
    batch = replay_memory.sample(3)
    _, _, _, s, _ = atari_collate(batch)
    assert s.shape == torch.Size([3, 4, 84, 84])

test_s_prime_shape_should_be_correct_give_atari_collate()

def test_not_terminated_shape_should_be_correct_give_atari_collate():
    batch = replay_memory.sample(3)
    _, _, _, _, t = atari_collate(batch)
    assert t.shape == torch.Size([3])

test_not_terminated_shape_should_be_correct_give_atari_collate()

def test_should_select_action_as_integer_given_epsilon_one():
    env = TorchEnv(gym.make(environment), transforms=pipeline)
    s = env.reset()
    print(s.shape)
    a = dqn.select_next_action(s.to(device), 1)
    assert isinstance(a, int)

test_should_select_action_as_integer_given_epsilon_one()

def test_should_assign_s_and_s_prime_same_shapes():
    env = TorchEnv(gym.make(environment), transforms=pipeline)
    s = env.reset()    
    s_prime, r, terminated = env.step(a)
    s_prime = torch.cat([s, s_prime])[-4:]
    assert s_prime.shape == s.shape

test_should_assign_s_and_s_prime_same_shapes()


def test_should_select_action_as_integer_given_epsilon_zero():
    env = TorchEnv(gym.make(environment), transforms=pipeline)
    s = env.reset()
    print(s.shape)
    a = dqn.select_next_action(s.squeeze(1).unsqueeze(0).to(device), 0)
    assert isinstance(a, int)

test_should_select_action_as_integer_given_epsilon_zero()
def test_should_assign_state_with_uint8_dtype():
    env = TorchEnv(gym.make(environment), transforms=pipeline)
    s = env.reset()    
    s_prime, r, terminated = env.step(a)
    s_prime = torch.cat([s, s_prime])[-4:]
    assert s_prime.dtype == torch.uint8

test_should_assign_state_with_uint8_dtype()
def test_should_assign_state_with_uint8_dtype():
    env = TorchEnv(gym.make(environment), transforms=pipeline)
    s = env.reset()    
    s_prime, r, terminated = env.step(a)
    s_prime = torch.cat([s, s_prime])[-4:]
    assert s_prime.dtype == torch.uint8

test_should_assign_state_with_uint8_dtype()