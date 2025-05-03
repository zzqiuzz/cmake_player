#pragma once
#include<iostream> 



enum class Level{
    Error = 0, 
    Warning, 
    Info
};

class Log{
    public:
        void setLevel(Level level);
        void showErrorLevel(const char* message);
        void showWarningLevel(const char* message);
        void showInfoLevel(const char* message);
        Log(){}
        Log(const int num) : lognum(num){ 
        };
        void pringLogFile(){
            std::cout << logFile << std::endl;
        }
    private:
        Level level = Level::Info;
        const char* logFile = " ";
        const int lognum = 0;
};