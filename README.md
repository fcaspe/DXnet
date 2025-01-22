# DXnet

This project is archived. For more recent Sound Matching, visit for instance
https://github.com/Sound2Synth/Sound2Synth


### A CNN to perform Yamaha DX7 patch regression from audio samples.

DXnet generates [Yamaha DX7](http://www.vintagesynth.com/yamaha/dx7.php) patches from real audio samples. The model provided here 'desynthesizes' sounds captured in a **2 second window @ 8Khz**.
The CNN regresses a vector of lenght 145, which codifies a Yamaha DX7 patch.

## Requirements
To train the networks and perform patch operations, the scripts use the [dx7pytorch](https://github.com/fcaspe/dx7pytorch) module, a Pytorch Dataset of DX7 sounds synthesized on-the-fly,
which also provides a collection of almost 30k Patches.

## Training script

The training script **train.py** trains the model from a DX7 patch collection, synthesizing instances of 2 seconds of audio sampled at 8Khz.
An ad-hoc loss function is used, which applies different weights to the patch parameters. 

To reduce the computational load during training, the model provided here has been trained with a reduced patch collection, only including those which are Algorithm 2 and 5.
Furthermore, only 4 MIDI notes are employed for training, using maximum velocity for each instance.

## Inference script

The provided inference script, **wave_to_patch_file.py** opens a specified .WAV file, crops the first 2 seconds, mixes and resamples it to a single channel and then performs inference
on the DXnet model, generating a patch file which can be loaded into a Yamaha DX7 using a [SYSEX](https://electronicmusic.fandom.com/wiki/System_exclusive) dump. The file is also compatible with several DX7 emulators, like [DEXED](https://github.com/asb2m10/dexed).
