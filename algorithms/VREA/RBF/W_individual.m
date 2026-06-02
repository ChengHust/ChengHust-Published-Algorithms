classdef W_individual
%INDIVIDUAL - The class of an individual.
%
%   This is the class of an individual. An object of INDIVIDUAL stores all
%   the properties including decision variables, objective values,
%   constraint violations, and additional properties of an individual.
%
% INDIVIDUAL properties:
%   dec         <read-only>     decision variables of the individual
%   obj         <read-only>     objective values of the individual
%   add         <read-only>     additional properties of the individual
%
% INDIVIDUAL methods:
%   INDIVIDUAL	<public>        the constructor, all the properties will be
%                               set when the object is creating
%   decs        <public>      	get the matrix of decision variables of the
%                               population
%   objs        <public>        get the matrix of objective values of the
%                               population
%   adds        <public>        get the matrix of additional properties of
%                               the population

    properties
        dec;        % Decision variables of the individual
        obj;        % Objective values of the individual
    end
    methods
        %% Constructor
        function obj = W_individual(Decs,Objs)
        
            if nargin > 0
                % Create new objects
                obj(1,size(Decs,1)) = W_individual;
                % Assign the decision variables, objective values and additional properties
                for i = 1 : length(obj)
                    obj(i).dec = Decs(i,:);
                    obj(i).obj = Objs(i,:);
                end
            end
        end
        %% Get the matrix of decision variables of the population
        function value = decs(obj)
        %decs - Get the matrix of decision variables of the population.
        %
        %   A = obj.decs returns the matrix of decision variables of the
        %   population obj, where obj is an array of INDIVIDUAL objects.
        
            value = cat(1,obj.dec);
        end
        %% Get the matrix of objective values of the population
        function value = objs(obj)
        %objs - Get the matrix of objective values of the population.
        %
        %   A = obj.objs returns the matrix of objective values of the
        %   population obj, where obj is an array of INDIVIDUAL objects.
        
            value = cat(1,obj.obj);
        end
    end
end