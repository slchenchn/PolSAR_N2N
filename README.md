# neighbor2neigbor for PolSAR data

# folder
`deprecated`: erroneous code, which is deprecated, but still valuable

# note
`model`: unetpp.py, 

# now
- without model summary
- without printing cfg to tensorboard
- without wrapped loss func
- the expectation value of logarithm of SAR intensity is not equal to logarithm of its expectation vallue, correlation is needed. See MuLoG, assume number of looks = 1

# TODO
- [x] rand_pool
- [x] test the PolSAR data loader
- [x] write the N2N loss
- [ ] debug the model, the loss value is too large, either log the Hoekman data or somethings else
- [ ] run multiple trails using .sh file