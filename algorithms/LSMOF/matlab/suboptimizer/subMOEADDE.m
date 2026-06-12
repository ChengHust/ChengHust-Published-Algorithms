function Population = subMOEADDE(Global,Population,Operator,Gmax)
%Sub-optimizer in CLEA (MOEA/D-DE)

        [delta,nr] = deal(0.9,2);
        [W,Global.N] = UniformPoint(Global.N,Global.M);
        B = pdist2(W,W);
        Z = min(Population.objs,[],1);
        [~,B] = sort(B,2);
        B = B(:,1:10);
        for gen = 1 : Gmax
            for i = 1 : Global.N
                if rand < delta
                    P = B(i,randperm(size(B,2)));
                else
                    P = randperm(Global.N);
                end
                if Operator == 1
                    Offspring = Global.Variation(Population([i,P(1:2)]),1,@DE);
                else
                    Offspring = Global.Variation(Population([i,P(1:2)]),1,@EAreal);
                end
                Z = min(Z,Offspring.obj);
                g_old = max(abs(Population(P).objs-repmat(Z,length(P),1)).*W(P,:),[],2);
                g_new = max(repmat(abs(Offspring.obj-Z),length(P),1).*W(P,:),[],2);
                Population(P(find(g_old>g_new,nr))) = Offspring;
            end
        end
end