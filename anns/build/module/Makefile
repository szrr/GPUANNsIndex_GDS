
ifneq ($(KERNELRELEASE),)
	src := /home/ErHa/GPUANNsIndex_GDS11.28/anns/bam/module
	obj-m := ANNSBaM.o
	ANNSBaM-objs := pci.o list.o ctrl.o map.o
	ccflags-y += -D_CUDA -I/usr/src/nvidia-550.90.12/nvidia -I/home/ErHa/GPUANNsIndex_GDS11.28/anns/bam/src -I/home/ErHa/GPUANNsIndex_GDS11.28/anns/bam/src/linux
	KBUILD_EXTRA_SYMBOLS := /usr/src/nvidia-550.90.12/Module.symvers
else

.PHONY: default reload unload load clean install

default:
	$(MAKE) -C /lib/modules/5.19.0-1010-nvidia-lowlatency/build M=/home/ErHa/GPUANNsIndex_GDS11.28/anns/build/module modules

clean:
	$(MAKE) -C /lib/modules/5.19.0-1010-nvidia-lowlatency/build M=/home/ErHa/GPUANNsIndex_GDS11.28/anns/build/module clean

reload: unload load

unload:
	-rmmod ANNSBaM.ko

load:
	insmod ANNSBaM.ko max_num_ctrls=64

install: default
	$(MAKE) -C /lib/modules/5.19.0-1010-nvidia-lowlatency/build M=/home/ErHa/GPUANNsIndex_GDS11.28/anns/build/module modules_install
	#$(MAKE) INSTALL_MOD_DIR=ANNSBaM -C /lib/modules/5.19.0-1010-nvidia-lowlatency/build M=/home/ErHa/GPUANNsIndex_GDS11.28/anns/build/module modules_install

endif

