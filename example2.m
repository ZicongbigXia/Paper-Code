
%Code for the RNN in Example 2 of "Z. Xia, et al, Distributed Nonconvex Optimization Subject to Globally Coupled Constraints via Collaborative Neurodynamic Optimization" submitted to Neural Networks  
 
step=0.01;
tend=1;
tspan= [0 tend];   
 
s=[1 1 2 2 4 5];
t=[2 5 3 4 5 6];
G = graph(s,t);
A= adjacency(G);
L =laplacian(G);
 cc=eye(6);
B=diag(cc(1,:));
for k=2:6
B=blkdiag(B,diag(cc(k,:)));
end
 up=[35 30 50 30 20 30]';
 low=[10 0 0 0 0 15]';
f0=zeros(54,1);
f0(1:6)=(up-low).*randi(10000,6,1)/10000+low-randi(100000,6,1)/1000;
 
 a=[0.001 0 0 0 0 0.002]'; 
 b=[-0.1 0 0 0 0 -0.16]';
 c=[3.699 1 1 1 1 4.698]';
 d=[-60.96 -41 -81 -51 -31 -60.9]'+1;

 H1=[1 1 1 1 1 1]';
 D=145;
 alpha1=0.1;
 
options=odeset('RelTol',1e-8,'AbsTol',1e-8);
[t,f]=ode15s(@(t,y) fun1(t,y,L,B,a,b,c,d,up,low,H1,D,alpha1),tspan,f0,options);
 plot(t,f(:,1:6),'linewidth',1.5) 
 plot(t,f(:,19:54),'linewidth',1.5) 
 f(end,1:6)
 sum(f(end,1:6))
function df=fun1(t,f,L,B,a,b,c,d,up,low,H1,D,alpha1)
x=f(1:6);
lamH1=f(7:12);
 wlamH1 =f(13:18);
  xx=f(19:54);
dfx=4*a.*x.^3+3*b.*x.^2+2*c.*x+d;
 BB=100;
CC=kron(L,eye(6));
%xx yy-layer

Dxx=-BB*sat(CC*xx+B*(xx-kron(ones(6,1),x)));
 
hh=[sum(xx(1:6));sum(xx(7:12));sum(xx(13:18));sum(xx(19:24));sum(xx(25:30));sum(xx(31:36))];  
 
Dx=max(low,min(up,x-dfx-lamH1.*H1-2*alpha1*hh.*H1))-x;
 %dlx-layer
 
DlamH1=H1.*x-ones(6,1)*(D/6)-L*lamH1-L*wlamH1;
DwlamH1=L*lamH1;
df=  500*[Dx;DlamH1;DwlamH1;Dxx];
end

