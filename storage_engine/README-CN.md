# SSD启用步骤
前置条件：按照英文版的README，确保软件环境和硬件环境符合要求
本文中所有的命令，凡最前面加了`~#`的代表需要`sudo -i`在超级用户下进行
未加`~#`的在普通用户下进行
## 将所需的nvmeSSD解挂：
### 确定设备名称：`lsblk`
部分结果如下：
``` bash
sda           8:0    0   3.7T  0 disk 
└─sda1        8:1    0   3.7T  0 part 
sdb           8:16   0   3.7T  0 disk 
└─sdb1        8:17   0   3.6T  0 part /mnt/data2
nvme1n1     259:0    0     7T  0 disk 
nvme0n1     259:1    0 931.5G  0 disk 
├─nvme0n1p1 259:2    0   512M  0 part /boot/efi
└─nvme0n1p2 259:3    0   931G  0 part /
```
我们要使用名为`nvme1n1`的7T SSD，这里非常重要，千万不能找错。
### 确定该SSD的PCIE设备地址：`dmesg | grep -i nvme1`
结果如下：
```bash
[    3.163998] nvme nvme1: pci function 0000:af:00.0
[    3.167432] nvme nvme1: Shutdown timeout set to 10 seconds
[    3.169627] nvme nvme1: 15/0/0 default/read/poll queues
```
我们需要的地址是`0000:af:00.0`
### 将这块盘从当前驱动程序中解绑
``` ~# echo -n "0000:af:00.0" > /sys/bus/pci/devices/0000\:af\:00.0/driver/unbind```
> Tips:解绑后，再执行`lsblk`就看不到nvme1这一块盘了。如果想要把他重新作为普通的硬盘来用，需要重新绑定到 nvme 驱动程序上：`~# echo -n "0000:af:00.0" > /sys/bus/pci/drivers/nvme/bind`

# 编译步骤
*在giann_core项目根目录下执行以下命令*
```bash
mkdir build && cd build
cmake ..
cd module
make
sudo make load
cd ..
make libnvm
make benchmarks
```
# 试运行
```
sudo ./bin/anns-bench
```