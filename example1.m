
%Code for the RNN in Example 1 of "Z. Xia, et al, Distributed Nonconvex Optimization Subject to Globally Coupled Constraints via Collaborative Neurodynamic Optimization" submitted to Neural Networks  

tend=0.5;% total time
tspan= [0 tend];   


%graph
s=[1 1 1 2 2 3 4 4 5 6 7];
t=[2 5 8 3 8 4 5 7 6 7 8];
G = graph(s,t);
A= adjacency(G);
L =laplacian(G);

%B-matrix
cc=eye(8);
B=diag(cc(1,:));
for k=2:8
B=blkdiag(B,diag(cc(k,:)));
end

%initial states
f0=zeros(240,1);
f0(1:8)=randi(10000,8,1)/1000-randi(10000,8,1)/1000;

% objective function and constraints
 a=[0.2 0.1 0.1 0.1 0.1 0.15 0.1 0.15]'; 
 b=[-2 -2 -2 -2 -2 -3 -2 -3]';
 c=[7 8 8 10 8 14 7 7]';
 d=[-9 -9 -7 -7 -5 -5 -5 -5]';
 up=15;
 low=0;
 H1=[1,0.5,1,0,0,0,0,1]';
 H2=[0,0,1,1,0,0,0,1]';
 b1=36;
 b2=35;
 
 %other parameters
 alpha2=0.1;
 gamma=1;
 
 options=odeset('RelTol',1e-8,'AbsTol',1e-8);% ODE paramters
 [t,f]=ode15s(@(t,y) fun1(t,y,L,B,a,b,c,d,up,low,H1,H2,b1,b2,alpha2,gamma),tspan,f0,options);

 plot(t,f(:,1:8),'linewidth',1.5) % curve for x
function df=fun1(t,f,L,B,a,b,c,d,up,low,H1,H2,b1,b2,alpha2,gamma)
x=f(1:8);
lx=f(9:16);
lamH1=f(17:24);
wlamH1=f(25:32);
lamH2=f(33:40);
wlamH2=f(41:48);
xx=f(49:112);
llamH1=f(113:176);
llamH2=f(177:240);
dfx=4*a.*x.^3+3*b.*x.^2+2*c.*x+d;
b=100;
CC=kron(L,eye(8));
%\hatx-layer
Dxx=-b*sat(CC*xx+B*(xx-kron(ones(8,1),x)));
%\hatlambda-layer
DllamH1=-b*sat(CC*llamH1+B*(llamH1-kron(ones(8,1),lamH1)));
DllamH2=-b*sat(CC*llamH2+B*(llamH2-kron(ones(8,1),lamH2)));
hhH1=llamH1.*(kron(ones(8,1),H1).*xx);
hhH1=hhH1([1 8+2 16+3 24+4 32+5 40+6 48+7 56+8]);
hhH2=llamH2.*(kron(ones(8,1),H2).*xx);
hhH2=hhH2([1 8+2 16+3 24+4 32+5 40+6 48+7 56+8]);
%x-layer
Dx=max(0,min(up,x-dfx-gamma*L*lx-gamma*L*x-lamH1.*H1-lamH2.*H2-2*alpha2*hhH1.*H1-2*alpha2*hhH2.*H2))-x;
%omega-layer
Dlx=gamma*L*x;
%lambda-layer
DlamH1=max(0,lamH1-H1.*x-b1-L*lamH1-L*wlamH1)-lamH1;
DlamH2=max(0,lamH2-H2.*x-b2-L*lamH2-L*wlamH2)-lamH2;
%\barlambda-layer
DwlamH1=L*lamH1;
DwlamH2=L*lamH2;
df=1000*[Dx;Dlx;DlamH1;DwlamH1;DlamH2;DwlamH2;Dxx;DllamH1;DllamH2];
end

