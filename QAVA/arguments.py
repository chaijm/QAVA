import argparse

def get_args():
    parser = argparse.ArgumentParser(description='A3C')
    parser.add_argument('--actor-lr', type=float, default=3e-4,
                        help='learning rate of actor network (default: 0.000001)')
    parser.add_argument('--critic-lr', type=float, default=3e-3,
                        help='learning rate of critic network (default: 0.00001)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for rewards (default: 0.9)')
    parser.add_argument('--gae-lambda', type=float, default=1.00,
                        help='lambda parameter for GAE (default: 1.00)')
    parser.add_argument('--entropy-coef', type=float, default=1,
                        help='entropy term coefficient (default: 1)')
    parser.add_argument('--value-loss-coef', type=float, default=0.5,
                        help='value loss coefficient (default: 0.5)')
    parser.add_argument('--max-grad-norm', type=float, default=50,
                        help='value loss coefficient (default: 50)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--num-processes', type=int, default=2,
                        help='how many training processes to use (default: 4)')
    parser.add_argument('--num-steps', type=int, default=10,
                        help='number of forward steps in A3C (default: 10)')
    parser.add_argument('--max-episode-length', type=int, default=1000000,
                        help='maximum length of an episode (default: 1000000)')
    parser.add_argument('--nn_model', default=None,
                        help='the path of the pre-trained model.')
    parser.add_argument('--model-save-interval', default=300,
                        help='save the model in how many epochs')
    parser.add_argument('--summary-dir', default=None,
                        help='the path to save model')
    parser.add_argument('--total-agents', default=11,
                        help='The maximum number of agents')
    parser.add_argument('--rand-range', default=1000,
                        help='used for random sample to determine action')
    parser.add_argument('--default_action', default=0,
                        help='the default index of initial action, the lowest bitrate')

    args = parser.parse_args()

    return args

