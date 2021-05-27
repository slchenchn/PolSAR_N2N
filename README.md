# neighbor2neigbor for PolSAR data

# folder
`deprecated`: erroneous code, which is deprecated, but still valuable

# note
`model`: unetpp.py, 

# now
- without model summary
- without printing cfg to tensorboard
- without wrapped loss func
- the expectation value of logarithm of SAR intensity is not equal to logarithm of its expectation vallue, correction is needed. See MuLoG, assume number of looks = 1
- turn off the check of assigned maximum should be less than real max value: `/home/csl/anaconda3/lib/python3.7/site-packages/piq/utils/common.py", line 54`, and set the allow_negative to True

# TODO
- [x] rand_pool
- [x] test the PolSAR data loader
- [x] write the N2N loss
- [x] debug the model, the loss value is too large, either log the Hoekman data or somethings else
- [x] run multiple trails using .sh file    `2021-05-26`
- [ ] accelarate the data loading time
- [ ] revise PSNR and SSIM, discard `data_range` param
- [ ] write inverse hoekman decomposition, show filtering result