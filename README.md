# Dial-a-Ride
## _Combinatorial Problems with Transformers_

Finding solutions to the Dial-a-Ride Problem with learned policies. The use of Transformers as policy model captures the complexity of the problems input. [See poster](https://github.com/Thibaud-Ardoin/Dial-a-Ride/blob/master/poster%20darp.pdf).

## Code

Different code elements are:
- Simulation environment.
- Dataset addaptation to simulator.
- Optimal supervision function (link to rf)
- Training Environement.
- Visual output.
- ClearML logging system.

<img src="https://user-images.githubusercontent.com/36546850/137620915-a5b39c13-55c1-4f0f-9964-fe148cd6c65a.gif" width="400" height="400"/>


## Usage

By running `dialRL/run_supervised_rl_clearML.py` a supervision train get started. All the desired parameters can be directly set as option in the bash call. From there a ClearML callback will save all the information needed to your own clearML account.

```bash
python dialRL/run_supervised_rl_clearML.py \
--total_timesteps 10000 \
--monitor_freq 1000 \
--example_freq 1000000 \
--epochs 10000 \
--alias TestSupervised \
--model Trans18 \
--layers 256 256 \
--eval_episodes 1 \
--verbose 0 \
--nb_target 16 \
--image_size 10 \
--nb_drivers 2 \
--dataset '' \
--reward_function ProportionalEndDistance \
--data_size 10000 \
--batch_size 2 \
--dropout 0 \
--lr 0.0001 \
--optimizer Adam \
--typ 33 \
--embed_size 512 \
--checkpoint_dir '' \
--vocab_size 16 \
--supervision rf \
--tag 'Local train' \
--heads 8 \
--forward_expension 4 \
--num_layers 6 \
--clearml 1 \
--balanced_dataset 0 \
--rl 0 \
--augmentation 0
```

# Results

On one hand, this Method manages to reproduce to 100% a simple sub-optimal strategy such as Nearest Neighbor. This enable us to conclude that our model fully capture the multitype constraints of the problem.

On the other hand, optimal strategies on small dataset are very complicated and are reproduced to only 92%. This keeps us a GAP to optimal strategy between 2 and 3%. This evaluations are made on Cordeau2007 dataset with instances that contain 3 drivers and 16 targets.

## Disclamer
The repo needs a bit of restructuration and cleaning and commenting here and there. But don't hesitate to ask for any question in the Issues, or per Mail.
 

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update pipeline as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
