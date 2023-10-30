defmodule BFS_Kernel do
  import GPotion

  gpotion kernel1(starting, no_of_edges, graph_edges, graph_mask, updating_graph_mask, graph_visited, cost, no_of_nodes, max_threads_per_block) do
    tid = blockIdx.x * max_threads_per_block + threadIdx.x
    if tid < no_of_nodes and graph_mask[tid] == 1 do
      graph_mask[tid] = 0
      for i in range(starting[tid], no_of_edges[tid] + starting[tid]) do
        id = graph_edges[i]
        if graph_visited[id] == 1 do
          cost[id] = cost[tid] + 1
          updating_graph_mask[id] = 1
        end
      end
    end
  end

  gpotion kernel2(graph_mask, updating_graph_mask, graph_visited, over, no_of_nodes, max_threads_per_block) do
    tid = blockIdx.x * max_threads_per_block + threadIdx.x
    if tid < no_of_nodes && updating_graph_mask[tid] == 1 do
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

input = case File.read("graph.txt") do
  {:ok, content} -> content |> String.split |> Enum.map(&String.to_integer/1)
  {:error, reason} -> exit(reason)
end

no_of_nodes = Enum.at(input, 0)

num_of_blocks = 1
num_of_threads_per_block = no_of_nodes

{num_of_blocks, num_of_threads_per_block} = if no_of_nodes > max_threads_per_block do
  {ceil(no_of_nodes/max_threads_per_block), max_threads_per_block}
end

nodes = Enum.slice(input, 1, no_of_nodes) |> Enum.chunk_every(2)

starting = for i <- nodes do Enum.at(i, 0) end
no_of_edges = for i <- nodes do Enum.at(i, 1) end
graph_mask = List.duplicate(0, no_of_nodes)
updating_graph_mask = List.duplicate(0, no_of_nodes)
graph_visited = List.duplicate(0, no_of_nodes)

source = 0

graph_mask = List.replace_at(graph_mask, source, 1)
graph_visited = List.replace_at(graph_visited, source, 1)

edge_list_size = Enum.at(input, no_of_edges + 1)
edges = Enum.slice(input, no_of_edges + 2, edge_list_size) |> Enum.chunk_every(2)
graph_edges = for i <- edges do Enum.at(i, 0) end
cost = List.last(edges) |> Enum.at(1)


m_starting = Matrex.new(starting)
m_no_of_edges = Matrex.new(no_of_edges)
m_graph_mask = Matrex.new(graph_mask)
m_updating_graph_mask = Matrex.new(updating_graph_mask)
m_graph_visited = Matrex.new(graph_visited)
m_graph_edges = Matrex.new(graph_edges)


copy_start = System.monotonic_time()

g_starting = GPotion.new_gmatrex(m_starting)
g_no_of_edges = GPotion.new_gmatrex(m_no_of_edges)
g_graph_mask = GPotion.new_gmatrex(m_graph_mask)
g_updating_graph_mask = GPotion.new_gmatrex(m_updating_graph_mask)
g_graph_visited = GPotion.new_gmatrex(m_graph_visited)
g_graph_edges = GPotion.new_gmatrex(m_graph_edges)

d_cost = List.duplicate(-1, no_of_nodes) |> List.replace_at(0, 0)
m_d_cost = Matrex.new(d_cost)
g_d_cost = GPotion.new_gmatrex(m_d_cost)

over = [0]
m_over = Matrex.new(over)
g_over = GPotion.new_gmatrex(m_over)

copy_end = System.monotonic_time()


k = 0
stop = 0

grid = {num_of_blocks, 1, 1}
threads = {num_of_threads_per_block, 1, 1}

kernel1 = GPotion.load(&BFS_Kernel.kernel1/10)
kernel2 = GPotion.load(&BFS_Kernel.kernel2/7)

exec = Stream.unfold(true, fn
  false ->
    IO.puts "Iterations: #{k}"
    nil
  true ->
    g_over = GPotion.new_gmatrex(m_over)
    GPotion.spawn(kernel1, grid, threads, [g_starting, g_no_of_edges, g_graph_edges, g_graph_mask, g_updating_graph_mask, g_graph_visited, g_d_cost, no_of_nodes, max_threads_per_block])
    GPotion.spawn(kernel2, grid, threads, [g_graph_mask, g_updating_graph_mask, g_graph_visited, g_over, no_of_nodes, max_threads_per_block])
    GPotion.synchronize()
    r = GPotion.get_gmatrex(g_over)
    k = k + 1
    b = if r[0] == 0, do: false, else: true
    {true, b}
end)

Stream.run(exec)

get_result_start = System.monotonic_time()
result = GPotion.get_gmatrex(g_d_cost)
get_result_end = System.monotonic_time()

File.write("results.txt", g_d_cost)

IO.puts "Copied to GPU: #{System.convert_time_unit(copy_end - copy_start, :native, :millisecond)}"
IO.puts "Copied result from GPU: #{System.convert_time_unit(get_result_end - get_result_start, :native, :millisecond)}"
