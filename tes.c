#include <stdio.h>


int main()
{

    FILE *f = fopen("v1_net", "wb");
if (!f)
{
    printf("Failed to open file\n");
    
}
    return 0;
}