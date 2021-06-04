#include <iostream>
#include "Hash.h"
#include "LinearOctree.h"


#ifdef WIN32
#include <tchar.h>
int _tmain(int argc, _TCHAR* argv[])
#else
int main(int argc, char* argv[])
#endif
{
    LinearOctree<int> lo(8,3);
    std::cout << "Program Ended : )" << std::endl;
	return 1;
}

