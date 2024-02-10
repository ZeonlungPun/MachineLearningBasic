%SOM FOR OOP 
classdef SOMnetwork<handle
    properties
        max_diameter%最大优胜领域
        min_diameter
        max_rate%学习率
        min_rate
        steps%迭代次数
        ratelist
        diameterlist
        x%数据集
        gridlocation%竞争层的神经元节点位置坐标
        w%神经元节点权重
        griddist%神经元节点之间的距离
    end
    methods(Static)   
        function gdist=calGdist(grid)
            m=size(grid,1)
            gdist=zeros(m,m)
            for i=1:m
                for j=1:m
                    if i~=j
                        gdist(i,j)=pdist2(grid(i,:),grid(j,:));
                    end
                end
            end
        end
        function grid=init_grid(m,n)%分成m*n类
            grid=zeros(m*n,2)
            k=1
            for i=1:m
                for j=1:m
                    grid(k,:)=[i,j]
                    k=k+1
                end
            end
        end    
    end
    methods
        function obj=SOMnetwork(max_diameter, min_diameter,max_rate, min_rate,steps)
            obj.max_diameter=max_diameter;
            obj.min_diameter=min_diameter;
            obj.max_rate=max_rate;
            obj.min_rate=min_rate;
            obj.steps=steps;
            obj.ratelist=[];
            obj.diameterlist=[];
            obj.x=0;
            obj.gridlocation=0;
            obj.w=0;
            obj.griddist=0;
        end
        function [rate,diameter]=changerate(obj,i)
            rate=obj.max_rate-(obj.max_rate-obj.min_rate).*(i+1)./obj.steps;
            diameter=obj.max_diameter-(obj.max_diameter-obj.min_diameter).*(i+1)./obj.steps;
        end
        function train(obj,x,m,n)%m,n输出层长度和宽度
            obj.x=x
            x=mapminmax(x',0,1);%对输入变量归一化
            x=x';
            [samples,features]=size(x);
            %初始化各节点位置以及各个节点之间的位置
            obj.gridlocation=init_grid(m,n);
            obj.griddist=calGdist(obj.gridlocation);
            %初始化各节点对应权值
            w=poslin(randn(m*n,features));
            if obj.steps<5*samples
                obj.steps=5*samples;
            end
            for i=1:obj.steps
            %竞争，随机选取样本计算距离
                no=randperm(samples);
                data=x(no(1),:);
                xdist=[];
                for j=1:size(w,1)
                    dd=pdist2(data,w(j,:));
                    xdist=[xdist;dd];
                end
                %找优胜点
                [~,winnerID]=min(xdist);
                %迭代，确定学习率和节点优胜半径，并保存
                [rate,diameter]=changerate(i,obj.steps,obj.max_rate,obj.min_rate,obj.max_diameter,obj.min_diameter);
                %圈定优胜领域所有节点
                judge=(obj.griddist(winnerID,:)<diameter);
                winnerroundID=find(judge~=0);
                w(winnerroundID,:)=  w(winnerroundID,:)+rate*(data-w(winnerroundID,:));
            end
            obj.w=w;
        end
        function label=cluster(obj,x)
            label=[];   
            x=mapminmax(x',0,1);%对输入变量归一化
            x=x';
            m=size(x,1);
            for i=1:m
                dist=sum((x(i,:)-obj.w).^2,2);
                [~,label1]=min(dist);
                label=[label;label1];
            end
        end
    end
end
    
    
     
        