using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using TensorFlow;
using TensorFlowSharp;

using System.Numerics;

namespace TensorFlowSharp
{
    class Program
    {



        static void Main(string[] args)
        {

            var g = new TFGraph();

            var sess = new TFSession(g);
            


            

            Program.Demo(g,sess);

            Program.First();

            Program.placeholder(g,sess);

            Program.Variable(g,sess);

            Program.InitVariable(g,sess);

            sess.CloseSession();

            Console.ReadKey();

        }

        public static void First()
        {
            Console.WriteLine("。。。。。。。。。。。。。。。。。测试。。。。。。。。。。。。。。。。。。");
             using (var session = new TFSession())
            {
                var graph = session.Graph;
                var a = graph.Const(2);
                var b = graph.Const(3);
                Console.WriteLine("a=2 b=3");
                var addingResults = session.GetRunner().Run(graph.Add(a, b));
                var addingResultsValue = addingResults.GetValue();
                Console.WriteLine("a+b={0}",addingResultsValue);
                var multiplyResults = session.GetRunner().Run(graph.Mul(a, b));
                var multiplyResultsValues = multiplyResults.GetValue();
                Console.WriteLine("a*b={0}", multiplyResultsValues);

      
            }
           
        }

        //占位符
        private static void placeholder(TFGraph g, TFSession sess)
        {

            Console.WriteLine("。。。。。。。。。。。。。。。。。占位符。。。。。。。。。。。。。。。。。。");
            //var g = new TFGraph();
            //var sess = new TFSession();

            var x = g.Placeholder(TFDataType.Float);
            var y = g.Placeholder(TFDataType.Float);
            var z = g.Placeholder(TFDataType.Float);


            var a1 = g.Add(x, y);
            var b1 = g.Mul(a1, z);
            var c1 = g.Pow(b1, g.Const(2.0f));
            var d1 = g.Div(c1, x);
            var e1 = g.Sqrt(d1);

            //var mn=

            var result3 = sess.GetRunner().AddInput(x, 1.0f)
                .AddInput(y, 2.0f)
                .AddInput(z, 3.0f)
                .Run(e1).GetValue();


            Console.WriteLine("e={0}", result3);

            //sess.CloseSession();
        }


        private static void Demo(TFGraph g,TFSession sess)
        {
            Console.WriteLine("。。。。。。。。。。。。。。。。。Demo。。。。。。。。。。。。。。。。。。");

            var a = g.Const(2);
            var b = g.Const(3);

            var add = g.Add(a, b);
            var mul = g.Mul(a, b);

            var result = sess.GetRunner().Run(add).GetValue();


            Console.WriteLine("a+b={0}", result);

            var result2 = sess.GetRunner().Run(mul).GetValue();

            Console.WriteLine("a*b={0}", result2);

        
        }

        //变量
        private static void Variable(TFGraph g, TFSession sess)
        {
            Console.WriteLine("。。。。。。。。。。。。。。Variable。。。。。。。。。。。。。。。。。。");
            TFOperation init;
            TFOutput value;
            var a2 = g.Variable(g.Const(1.5), out init, out value);

            var inc = g.Const(0.5);

            var update = g.AssignVariableOp(a2, g.Add(value, inc));


            sess.GetRunner().AddTarget(init).Run();

            for (var i = 0; i < 5; i++)
            {
                var result4 = sess.GetRunner().Fetch(value).AddTarget(update).Run();
                Console.WriteLine("result{0}:{1}", i, result4[0].GetValue());
            }
        }


        //初始化
        private static void InitVariable(TFGraph g, TFSession sess)
        {
            Console.WriteLine("。。。。。。。。。。。。。。。。。初始化。。。。。。。。。。。。。。。。。。");
            TFStatus status = new TFStatus();

            var a = g.VariableV2(TFShape.Scalar,TFDataType.Double);

            var initA = g.Assign(a,g.Const(1.5));

            var b= g.VariableV2(new TFShape(99), TFDataType.Int32);

            //var initB = g.Assign(b, g.Range(g.Const(1),g.Const(5)));
            var initB = g.Assign(b, g.Range(g.Const(1), g.Const(100)));
            var run = sess.GetRunner();

            run.AddTarget(initA.Operation,initB.Operation).Run(status);
            Console.WriteLine(status.StatusCode);

            var res = run.Fetch(a,b).Run();

            Console.WriteLine(res[0].GetValue());
            Console.WriteLine(string.Join(",",(int[])res[1].GetValue()));
        }
    }
}
