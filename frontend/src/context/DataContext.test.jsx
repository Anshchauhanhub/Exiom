import React, { useState } from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom/vitest';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { DataProvider, useData } from './DataContext';

const { listGoalsMock, getRoadmapMock, deleteGoalMock } = vi.hoisted(() => ({
  listGoalsMock: vi.fn(),
  getRoadmapMock: vi.fn(),
  deleteGoalMock: vi.fn(),
}));

vi.mock('./AuthContext', () => ({
  useAuth: () => ({ user: { id: 'user-1' } }),
}));

vi.mock('../services/api', () => ({
  listGoals: listGoalsMock,
  getRoadmap: getRoadmapMock,
  toggleGoalStatus: vi.fn(),
  deleteGoal: deleteGoalMock,
}));

const goal = { id: 'goal-1', title: 'Test learning path', status: 'active' };

const DeleteGoalProbe = () => {
  const { goals, deleteGoalOptimistic } = useData();
  const [error, setError] = useState('');

  const handleDelete = async () => {
    try {
      await deleteGoalOptimistic(goal.id);
    } catch {
      setError('Delete failed');
    }
  };

  return (
    <>
      {goals.map(item => <p key={item.id}>{item.title}</p>)}
      <button onClick={handleDelete}>Delete path</button>
      {error && <p role="alert">{error}</p>}
    </>
  );
};

describe('DataProvider goal deletion', () => {
  beforeEach(() => {
    localStorage.clear();
    vi.clearAllMocks();
    listGoalsMock.mockResolvedValue([goal]);
    getRoadmapMock.mockResolvedValue({ tasks: [] });
    deleteGoalMock.mockResolvedValue({ message: 'Goal deleted', id: goal.id });
  });

  it('removes a goal after the delete request succeeds', async () => {
    render(
      <DataProvider>
        <DeleteGoalProbe />
      </DataProvider>
    );

    await screen.findByText(goal.title);
    fireEvent.click(screen.getByRole('button', { name: 'Delete path' }));

    await waitFor(() => expect(screen.queryByText(goal.title)).not.toBeInTheDocument());
    expect(deleteGoalMock).toHaveBeenCalledWith(goal.id);
  });

  it('restores a goal when the delete request fails', async () => {
    deleteGoalMock.mockRejectedValue(new Error('Request failed'));
    render(
      <DataProvider>
        <DeleteGoalProbe />
      </DataProvider>
    );

    await screen.findByText(goal.title);
    fireEvent.click(screen.getByRole('button', { name: 'Delete path' }));

    expect(await screen.findByRole('alert')).toHaveTextContent('Delete failed');
    expect(screen.getByText(goal.title)).toBeInTheDocument();
  });
});
