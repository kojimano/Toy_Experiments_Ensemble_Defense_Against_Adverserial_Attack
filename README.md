# LeNet-5

This implements a slightly modified LeNet-5 [LeCun et al., 1998a] and achieves an accuracy of ~99% on the [MNIST dataset](http://yann.lecun.com/exdb/mnist/). The Original code was taken from https://github.com/activatedgeek/LeNet-5.


![Epoch Train Loss visualization](https://i.imgur.com/h4h7CrF.gif)

## Setup

Install all dependencies using the following command

```
$ pip install -r requirements.txt
```

## Usage

Start the training procedure of LeNet for training clones to be used in Adeverserial Attacks. We trained 20 clones to split them 10 and 10 into defender ensemble and attacker ensemble networks.

```
$ python run.py --num_clones 20
```

After training LeNet clones, you can simulate an attacker and a defender using different options in MNIST toy examples. In the default setting, we uses ensemble of 10 clones foe non-target adverserial attacks [1, 3], and the random sampling of 10 clones for multiplication logic for a defender network. To run this, use
```
$ python Adverserial.py --num_attacker_clone 10 --num_defender_clone 10
```
We also allow users to try out different ensemble logics and different sampling logics for a defender network. Please see the detailed result in our technical report [1].    

See epoch train loss live graph at [`http://localhost:8097`](http://localhost:8097).

## References
[[1](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)] Our Technical Report
[[2](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based learning applied to document recognition." Proceedings of the IEEE, 86(11):2278-2324, November 1998.
[[3](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)]
