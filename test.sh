echo "SIMPLE(MULT) / Ensemble"
python Adversarial.py --num_attacker_clone 10 --num_defender_clone 10 --defender_sample single
echo "Ensemble(MULT) / Ensemble"
python Adversarial.py --num_attacker_clone 10 --num_defender_clone 10 --defender_sample all
echo "Random Sample(MULT) /Ensemble"
python Adversarial.py --num_attacker_clone 10 --num_defender_clone 10 --defender_sample random

