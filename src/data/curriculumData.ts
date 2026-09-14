import { Module } from '../types/curriculum';
import { foundationModules } from './modules/foundations';
import { ragModules } from './modules/ragModules';
import { agentModules } from './modules/agentModules';
import { systemsModules } from './modules/systemsModules';
import { capstoneAndPracticeModules } from './modules/capstoneAndPractice';

export const curriculumData: Module[] = [
  ...foundationModules,
  ...ragModules,
  ...agentModules,
  ...systemsModules,
  ...capstoneAndPracticeModules
];
