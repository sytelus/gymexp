for i in {1..5}
do
  python frozenlake_dqn_agent.py --density $i --slippery 0 --algo dqn
done