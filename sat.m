function M=sat(x)
k=100;
d=abs(x/k);
if d<=1
M=x/k;
else
M=sign(x/k);
end