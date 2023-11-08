#ifndef MPU6050_H
#define MPU6050_H

#include<stdint.h>
extern int fd;

int MPU6050_init();
uint8_t i2c_write(int fd,uint8_t reg,uint8_t val);
uint8_t i2c_read(int fd,uint8_t reg,uint8_t* val);
short GetData(int fd,unsigned char REG_Address);

#endif