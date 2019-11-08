# Mouse CNN project notes
#### notations and assumptions
Somewhat <font color='red'>arbituray choices</font> are color coded in <font color='red'>red</font>. Somewhat <font color='green'>natural choices</font> are color coded in <font color='green'>green</font>.
- input size $W$: <font color='green'>size_x = size_y = $W$ </font>
- kernel size $K$: <font color='red'>2x Gaussian width of connectivity estimation</font> 
- stride $S$: <font color='red'>fixed to be 1</font>
- output size shrinkage $\sigma$: <font color='red'>fixed to be 1</font>
- padding size $P$: calculated by relation $(W-K+2P)/S+1=\sigma W$, $P=\frac{(\sigma S-1)W+K-S}{2}$
- Input to same target area are <font color='red'>added</font> together
 
#### Conv2d parameters 
> torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros


- *in_channels*: # out_channels of source layer
- *out_channels*: # neurons in the area/output size$^2$ 
- *kernel_size*: <font color='red'>2x</font> Gaussian width 
- *stride*: <font color='red'>fixed to be 1</font>
- *padding*: calculated above
- *dilation, groups etc.*: use default


### Architecture
![](https://i.imgur.com/ILoDjcv.png)




