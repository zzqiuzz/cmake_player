#include "include/log.h" 
#include "include/entity.h"
#include<string>
using std::string;
int g_variable = 6;

class Engine {
public:
    // Engine 类有一个带参数的构造函数
    Engine(int power) : power(power) {}
    void start() const {
        std::cout << "Engine with " << power << " horsepower is starting." << std::endl;
    }
private:
    int power;
};

class Car {
public:
    // 构造函数使用初始化列表来初始化 const 成员变量
    Car(const std::string& model, int engine_power, int year)
        : model(model), engine(engine_power), year_built(year){
            // year_built = 13;
        } // 初始化列表

    void start_engine() const {
        engine.start();
        std::cout << "Built in year: " << year_built << std::endl;
    }

private:
    std::string model;
    Engine engine; // 成员变量 engine 是 Engine 类的对象
    const int year_built; // const 成员变量，必须在初始化列表中初始化
};

struct data{
    static int x;
    static int y;
    static void print(){
        std::cout << x << "," << y << std::endl;
    }
}; 
int data::x; // must initialized outside struct.
int data::y;
void increment(){
    static int i = 0;
    i++;
    std::cout << i << std::endl;
}


int main(){  

    Car my_car(string("Sedan"), 150, 2025);
    my_car.start_engine(); 

    // int x = 8;
    // int y = 9; 
    // const int* c = &x;
    // c = &y; 
    // auto f = [&x, &y](int a, int b){ 
    //     x++;
    //     a = b + x;
    //     std::cout << x << std::endl;
    // };     
    // f(x,y);
    // std::cout << x << std::endl;

    // const char* hello = "hello";
    // // hello[2] = 'f';
    // std::cout << hello << std::endl;
    // int array[5] = {0}; 
    // int* ptr = array;
    // *(ptr + 3) = 3;
    // std::cout << *(int*)((char*)ptr + 4 * 3) << std::endl;
    // std::cout << "---------------------------" << std::endl;
    // Entity e(1.0, 2.0);
    // e.print();
    // e.getX();
    // e.getY();
    // printClassName(&e);
    // Player p("lepaulski", 3.f, 4.f);
    // p.print();
    // p.getX();
    // p.getY();
    // printClassName(&p);
    // Entity* eptr = new Player("hh", 1, 3);
    // eptr->print();
    // printEntity(eptr);
    // static_cast<Entity*>(eptr)->Entity::print();
    // delete eptr;
    // std::cout << "---------------------------end" << std::endl;

    // data d0;
    // d0.x = 1;
    // d0.y = 1;
    // d0.print();
    // data::x = 10;
    // data::y = 10; 
    // d0.print();
    // increment();
    // increment();
    // increment();
    // int i = 10;
    // increment();
    // increment();

    // Log log0;
    // log0.pringLogFile();
    // Log log;
    // Level level = Level::Error;
    // log.setLevel(level);
    // log.showErrorLevel("Error");
    // level = Level::Warning;
    // log.setLevel(level);
    // log.showWarningLevel("Warning");
    // level = Level::Info;
    // log.setLevel(level);
    // log.showInfoLevel("Info");
    
}