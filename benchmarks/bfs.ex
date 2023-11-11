defmodule BFS_Kernel1 do
  import GPotion

  gpotion kernel1(starting, no_of_edges, graph_edges, graph_mask, updating_graph_mask, graph_visited, cost, no_of_nodes, max_threads_per_block) do
    tid = blockIdx.x * max_threads_per_block + threadIdx.x
    if tid < no_of_nodes && __float2int_rn(graph_mask[tid]) == 1 do
      graph_mask[tid] = 0
      s =  __float2int_rn(starting[tid])
      e =  __float2int_rn(no_of_edges[tid] + starting[tid])
      for i in range(s, e) do
        id =  __float2int_rn(graph_edges[i])
        if __float2int_rn(graph_visited[id]) == 0 do
          cost[id] = cost[tid] + 1
          updating_graph_mask[id] = 1
        end
      end
    end
  end
end

defmodule BFS_Kernel2 do
  import GPotion

  gpotion kernel2(graph_mask, updating_graph_mask, graph_visited, over, no_of_nodes, max_threads_per_block) do
    tid = blockIdx.x * max_threads_per_block + threadIdx.x
    if tid < no_of_nodes && __float2int_rn(updating_graph_mask[tid]) == 1 do
      graph_mask[tid] = 1
      graph_visited[tid] = 1
      over[0] = 1
      updating_graph_mask[tid] = 0
    end
  end
end


max_threads_per_block = 512

# struct Node
#   int starting
#   int no_of_edges

[file_input] = System.argv()

input_read_start = System.monotonic_time()

input = case File.read(file_input) do
  {:ok, content} -> content |> String.split |> Enum.map(&String.to_integer/1)
  {:error, reason} -> exit(reason)
end

no_of_nodes = Enum.at(input, 0)

num_of_blocks = 1
num_of_threads_per_block = no_of_nodes

{num_of_blocks, num_of_threads_per_block} = if no_of_nodes > max_threads_per_block do
  {ceil(no_of_nodes/max_threads_per_block), max_threads_per_block}
end

nodes = Enum.slice(input, 1, no_of_nodes * 2) |> Enum.chunk_every(2)

starting = for i <- nodes do Enum.at(i, 0) end
no_of_edges = for i <- nodes do Enum.at(i, 1) end
graph_mask = List.duplicate(0, no_of_nodes)
updating_graph_mask = List.duplicate(0, no_of_nodes)
graph_visited = List.duplicate(0, no_of_nodes)

source = Enum.at(input, no_of_nodes * 2 + 1)
source = 0

graph_mask = List.replace_at(graph_mask, source, 1)
graph_visited = List.replace_at(graph_visited, source, 1)

edge_list_size = Enum.at(input, no_of_nodes * 2 + 2)
edges = Enum.slice(input, no_of_nodes * 2 + 3, edge_list_size * 2) |> Enum.chunk_every(2)
graph_edges = for i <- edges do Enum.at(i, 0) end
cost = List.last(edges) |> Enum.at(1)

input_read_end = System.monotonic_time()
IO.puts "Input read time: #{System.convert_time_unit(input_read_end - input_read_start, :native, :millisecond)}ms"


matrex_new_start = System.monotonic_time()

m_starting = Matrex.new([starting])
m_no_of_edges = Matrex.new([no_of_edges])
m_graph_mask = Matrex.new([graph_mask])
m_updating_graph_mask = Matrex.new([updating_graph_mask])
m_graph_visited = Matrex.new([graph_visited])
m_graph_edges = Matrex.new([graph_edges])
over = [0]
m_over = Matrex.new([over])
d_cost = List.duplicate(-1, no_of_nodes) |> List.replace_at(0, 0)
m_d_cost = Matrex.new([d_cost])

matrex_new_end = System.monotonic_time()
IO.puts "Matrex init time: #{System.convert_time_unit(matrex_new_end - matrex_new_start, :native, :millisecond)}ms"


copy_start = System.monotonic_time()

g_starting = GPotion.new_gmatrex(m_starting)
g_no_of_edges = GPotion.new_gmatrex(m_no_of_edges)
g_graph_mask = GPotion.new_gmatrex(m_graph_mask)
g_updating_graph_mask = GPotion.new_gmatrex(m_updating_graph_mask)
g_graph_visited = GPotion.new_gmatrex(m_graph_visited)
g_graph_edges = GPotion.new_gmatrex(m_graph_edges)
g_d_cost = GPotion.new_gmatrex(m_d_cost)
g_over = GPotion.new_gmatrex(m_over)

copy_end = System.monotonic_time()
IO.puts "Copied to GPU: #{System.convert_time_unit(copy_end - copy_start, :native, :millisecond)}ms"


grid = {num_of_blocks, 1, 1}
threads = {num_of_threads_per_block, 1, 1}

kernel1 = GPotion.load(&BFS_Kernel1.kernel1/10)
kernel2 = GPotion.load(&BFS_Kernel2.kernel2/7)

kernel_exec_start = System.monotonic_time()

iter = Stream.iterate(0, &(&1 + 1))
  |> Enum.reduce_while(0, fn _i, acc ->
    g_over = GPotion.new_gmatrex(m_over)
    GPotion.spawn(kernel1, grid, threads, [g_starting, g_no_of_edges, g_graph_edges, g_graph_mask, g_updating_graph_mask, g_graph_visited, g_d_cost, no_of_nodes, max_threads_per_block])
    GPotion.spawn(kernel2, grid, threads, [g_graph_mask, g_updating_graph_mask, g_graph_visited, g_over, no_of_nodes, max_threads_per_block])
    GPotion.synchronize()
    stop = GPotion.get_gmatrex(g_over)[1] != 0

    if !stop do
      {:halt, acc}
    else
      {:cont, acc + 1}
    end
  end)

iter = iter + 1

kernel_exec_end = System.monotonic_time()
IO.puts "Kernel execution time: #{System.convert_time_unit(kernel_exec_end - kernel_exec_start, :native, :millisecond)}ms"


get_result_start = System.monotonic_time()
result = GPotion.get_gmatrex(g_d_cost)
get_result_end = System.monotonic_time()
IO.puts "Copied result from GPU: #{System.convert_time_unit(get_result_end - get_result_start, :native, :millisecond)}ms"

File.mkdir_p!("benchmarks/data/output/bfs")
Enum.with_index(result)
  |> Enum.reduce("", fn(r,string) -> string <> "#{elem(r,1)}) cost:#{trunc(elem(r,0))}\n" end)
  |> (&File.write("benchmarks/data/output/bfs/bfs_output.txt", &1)).()


IO.puts "Iterations: #{iter}"
