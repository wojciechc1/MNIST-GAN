GAN



 ![plot_avg_loss](metrics/d-1x_1000t_100e.png) | ![plot_avg_loss](metrics/d-2x_1000t_100e.png) | ![plot_avg_loss](metrics/d-3x_1000t_100e.png) |
|--------------------------|--------------------------------------|--------------------------------------|


### D trained **every** iteration:

- D performs too well (low D loss).
- G can’t keep up (high G loss).

### D trained **2×** less frequently:

- The generator still struggles to catch up (G loss is higher than D loss).

### D trained **3×** less frequently:

- Possibly a much better result (G loss is lower than D's).
- However, the trend suggests a role reversal, which is not desirable.

### Summary:
Training the D too often hinders the G’s laearning. Training it too rarely weakens its ability to distinguish. The best results come from balancing the learning pace of both networks.



#### 📈 Legend 

| Symbol | Meaning               |
|--------|------------------------|
| 🟥 Red line   | Discriminator loss      |
| 🟦 Blue line  | Generator loss          |
| **D**        | Discriminator            |
| **G**        | Generator                |
| 📊           | Average loss per epoch   |


 ![plot_avg_loss](metrics/d-3x_1000t_100e.png) | ![plot_avg_loss](metrics/d-3x_10000t_100e.png) |
|--------------------------|------------------------------------------------|

#### The plot flips here, indicating the G started to fail. I think that best performance was likely around 35 iterations (see second plot).