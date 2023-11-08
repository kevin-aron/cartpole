#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<math.h>
#include"bpnn.h"
#include"mpu.h"

#define LEFT_LIMIT 0
#define RIGHT_LIMIT 2000
#define THETA_LIMIT 0.2

#define FORCE 10
#define GRAVITY 9.8
#define CART_MASS 1.0
#define POLE_MASS 0.1
#define POLE_LEN 0.5

#define MaxMemory 100000
#define BATCH_SIZE 1000

#define ACCEL_XOUT_H                                0x3B
#define ACCEL_XOUT_L                                0x3C
#define ACCEL_YOUT_H                                0x3D
#define ACCEL_YOUT_L                                0x3E
#define ACCEL_ZOUT_H                                0x3F
#define ACCEL_ZOUT_L                                0x40

#define GYRO_XOUT_H                                 0x43
#define GYRO_XOUT_L                                 0x44
#define GYRO_YOUT_H                                 0x45
#define GYRO_YOUT_L                                 0x46
#define GYRO_ZOUT_H                                 0x47
#define GYRO_ZOUT_L                                 0x48

typedef struct{
    float pos_x;
    float pos_y;
    float pos_z;
    float acce_x;
    float acce_y;
    float acce_z;
} restate;

typedef struct{
    state nows;
    state nexts;
    int action;
    float reward;
} Experience;

restate s_now;
static state s;
Experience memory[MaxMemory];
int memory_size=0;

void set_state(float x,float v,float t,float w){
    s.pos=x;
    s.speed=v;
    s.theta=t;
    s.omega=w;
}
void set_restate(float px,float py,float pz,float ax,float ay,float az){
    s_now.pos_x=px;
    s_now.pos_y=py;
    s_now.pos_z=pz;
    s_now.acce_x=ax;
    s_now.acce_y=ay;
    s_now.acce_z=az;
}
void read_number(float *px,float *py,float *pz,float *ax,float *ay,float *az,int fd){
    usleep(1000 * 200);
    *ax=(float)GetData(fd,ACCEL_XOUT_H);
    usleep(1000 * 200);
    *ay=(float)GetData(fd,ACCEL_YOUT_H);
    usleep(1000 * 200);
    *az=(float)GetData(fd,ACCEL_ZOUT_H);
    usleep(1000 * 200);
    *px=(float)GetData(fd,GYRO_XOUT_H);
    usleep(1000 * 200);
    *py=(float)GetData(fd,GYRO_YOUT_H);
    usleep(1000 * 200);
    *pz=(float)GetData(fd,GYRO_ZOUT_H);
    sleep(1);
}
int out_limits(){
    if(s.pos>RIGHT_LIMIT||s.pos<LEFT_LIMIT||fabs(s.theta)>THETA_LIMIT) return 0;
    return 1;
}
void update_state(int flag){
    float force=(flag==1?1.0:-1.0)*10.0;
    float ct,st,dt=1;//时间步长
    float x_acc,t_acc;
    ct=cos(s.theta);
    st=sin(s.theta);
    t_acc=((CART_MASS+POLE_MASS)*GRAVITY*st-(force+POLE_MASS*POLE_LEN*s.omega*s.omega*st)*ct)/(4.0/3.0*(CART_MASS+POLE_MASS)*POLE_LEN-POLE_MASS*POLE_LEN*ct*ct);
    x_acc=(force+POLE_MASS*POLE_LEN*(s.omega*s.omega*st-t_acc*ct))/(POLE_MASS+CART_MASS);
    s.pos+=s.speed*dt;
    s.speed+=x_acc*dt;
    s.theta+=s.omega*dt;
    s.omega+=t_acc*dt;
}
void display(int fd){
    printf("ACCE_X:%f\n ", s_now.acce_x);
    printf("ACCE_Y:%f\n ", s_now.acce_y);
    printf("ACCE_Z:%f\n ", s_now.acce_z);
    printf("GYRO_X:%f\n ", s_now.pos_x);
    printf("GYRO_Y:%f\n ", s_now.pos_y);
    printf("GYRO_Z:%f\n\n ", s_now.pos_z);
}
int main(){
    BPNN bpnn;
    init_bpnn(&bpnn);
    initialize_weights(&bpnn);
    int fail=0;
    int episode=10000;
    srand(time(NULL));
    int fd=MPU6050_init();
    if(fd==-1) return 0;
    usleep(1000*100);
    float px,py,pz,ax,ay,az;
    for(int i=0;i<episode;i++){
        fail=0;
        int total_reward=0;
        int cnt=0;
        while(!fail){
            //获取陀螺仪读数
            read_number(&px,&py,&pz,&ax,&ay,&az,fd);
            display(fd);
            set_restate(px,py,pz,ax,ay,az);
            //将读数赋值给state
            s.pos=s_now.pos_x;
            s.speed=s_now.acce_x;
            //通过policy策略选择动作
            int action=greedypolicy(&bpnn,s);
            //放入经验池
            Experience exp;
            exp.action=action;
            exp.nows=s;
            //state temps=s;
            update_state(action);
            exp.nexts=s;
            //s=temps;
            fail=out_limits();
            if(!fail) total_reward+=1.0;
            else{
                total_reward-=1000.0;
                //回到重生瞄点
                set_state(0,0,0,0);
            }
            exp.reward=total_reward;
            memory[memory_size++]=exp;
            //训练模型
            if(memory_size>=BATCH_SIZE){
                int idx=rand()%memory_size;
                double input_data[4]={memory[idx].nows.pos,memory[idx].nows.speed,memory[idx].nows.theta,memory[idx].nows.omega};
                double target_left=memory[idx].reward;
                double target_right=memory[idx].reward;
                double target_data[2]={target_left,target_right};
                learn(&bpnn,input_data,target_data);
            }
        }
    }
    close(fd);
    cleanup_bpnn(&bpnn);
    printf("successful!\n");
    return 0;
}