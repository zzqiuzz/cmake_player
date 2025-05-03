#include<iostream>

class Example{
    public:
        Example(){
            std::cout << "Created Example." << std::endl;
        }
        Example(int a) : _a(a){
            std::cout << "Created Example" << _a << std::endl;
        }

    private:
        int _a;

};



class Printable{
    public:
        virtual void getClassName() = 0; // require any subclass inheriting from it implement getClassName
};

class Entity : public Printable{
    private:
        float _x;
        float _y;
        Example e;
    public:
        Entity(float x, float y): _x(x), _y(y), e(Example(5)){ 
            // Example(5);
        }
        Entity() = delete;
        virtual void print();
        void getClassName() override{
            std::cout << "Class Name is Entity." << std::endl;
        }
        void getX(){
            std::cout << "----_x" << _x << std::endl;
        };
        void getY(){
            std::cout << "----_y" << _y << std::endl;
        };
        ~Entity(){ 
        }
};

class Player : public Entity{
    private:
        const char* _name;
    public:
        Player(const char* name, float x, float y): Entity(x, y), _name(name){}
        void print() override;
        void getClassName() override{
            std::cout << "Class Name is Player." << std::endl;
        }
};

void printEntity(Entity* e);
void printClassName(Printable* p);