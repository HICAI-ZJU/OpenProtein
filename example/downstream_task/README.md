# Downstream Tasks Examples

This folder contains several use cases for downstream task models. These use cases mainly show the most common ways to call the downstream task models in OpenProtein.

To avoid errors, you need to install OpenProtein before calling these models. Then download the weights of the pre-trained models and the downstream task dataset to your local computer. Finally replace the paths in the above `py` files with `local paths` before running them.

You can refer to them to learn how to call these models in OpenProtein.

- Take the EC dataset as an example.

Assume that the pre-trained model parameters are stored in the `./checkpoints` folder. You can run the following command to test the effect of the protein sequence

```bash
python ec.py -s RLQIEAIVEGFTQMKTDLEKEQRSMASMWKKREKQIDKVLLNTTYMYGSIKGIAGNAVQTVSLLELPVDENGEDE -p ./checkpoints/esm_msa1b_t12_100M_UR50S.pt
```
