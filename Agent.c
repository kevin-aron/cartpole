#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<math.h>
#include"bpnn.h"

#define LEFT_LIMIT 0
#define RIGHT_LIMIT 2000
#define THETA_LIMIT 0.2

#define FORCE 10
#define GRAVITY 9.8
#define CART_MASS 1.0
#define POLE_MASS 0.1
#define POLE_LEN 0.5

#define MaxMemory 100000
#define BATCH_SIZE 10000

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
void read_number(float *px,float *py,float *pz,float *ax,float *ay,float *az){
    //*px=1.0;
}
int out_limits(){
    if(s.pos>RIGHT_LIMIT||s.pos<LEFT_LIMIT||fabs(s.theta)>THETA_LIMIT) return 0;
    return 1;
}
void update_state(int flag){
    float force=(flag==1?1.0:-1.0)*10.0;
    float ct,st,dt=0.1;//时间步长
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
int main(){
    BPNN bpnn;
    init_bpnn(&bpnn);
    initialize_weights(&bpnn);
    int fail=0;
    int episode=10000;
    srand(time(NULL));
    float px,py,pz,ax,ay,az;
    for(int i=0;i<episode;i++){
        read_number(&px,&py,&pz,&ax,&ay,&az);
        set_restate(px,py,pz,ax,ay,az);
        fail=0;
        int total_reward=0;
        while(!fail){
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

    cleanup_bpnn(&bpnn);
    printf("successful!\n");
    return 0;
}